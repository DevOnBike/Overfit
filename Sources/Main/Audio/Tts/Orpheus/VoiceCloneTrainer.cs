// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Optimizers;

namespace DevOnBike.Overfit.Audio.Tts.Orpheus
{
    /// <summary>
    /// QLoRA fine-tunes Orpheus on a target voice — the training half of voice cloning, pure managed .NET. The
    /// frozen 4-bit base is never modified; a small LoRA adapter learns to emit the target speaker's audio tokens
    /// for given text. Feed it <see cref="OrpheusTrainingExample"/>s from
    /// <see cref="VoiceCloneDatasetBuilder"/>; the loss is completion-only (prompt masked). Save the adapter, then
    /// load it into an Orpheus engine to speak in the new voice.
    /// <para>
    /// Memory note: audio-token sequences are long and Orpheus's vocab is large (~156 k), so the logits arena
    /// dominates — keep clips short (≈1–3 s) and size <c>maxSeqLen</c> to your longest example.
    /// </para>
    /// </summary>
    public sealed class VoiceCloneTrainer : IDisposable
    {
        private const int IgnoreIndex = -1;

        private readonly QLoRAOptions _opt;
        private readonly CachedLlamaInferenceEngine _engine;
        private readonly TrainableLlamaModel _model;
        private readonly ComputationGraph _graph;
        private readonly int _vocab;          // the LM-head/loss output size (restricted to audio tokens)
        private readonly int _outputStart;    // real token id of restricted index 0 (0 when not restricted)
        private readonly int _maxSeqLen;

        private readonly bool _useCheckpoint;

        /// <summary>
        /// Loads the Orpheus base + builds a fresh trainable adapter. <paramref name="useCheckpoint"/> recomputes
        /// each block's forward during the backward pass to keep only one block's activations live — <b>required
        /// for this 28-layer / 3 B model</b> (disabling it makes the full-graph activations exceed the arena and
        /// OOMs, measured), so it defaults on; the flag is kept only for small models whose whole graph fits.
        /// </summary>
        public VoiceCloneTrainer(string orpheusGgufPath, int maxSeqLen = 768, QLoRAOptions? options = null,
            bool useCheckpoint = true, bool restrictToAudioVocab = true)
        {
            _opt = options ?? new QLoRAOptions();
            _maxSeqLen = maxSeqLen;
            _useCheckpoint = useCheckpoint;
            _engine = CachedLlamaInferenceEngine.LoadGguf(orpheusGgufPath);
            Tokenizer = new GgufEmbeddedTokenizer(GgufTokenizer.Load(orpheusGgufPath));

            // Restrict the LM head / loss / generation to the audio-token sub-vocabulary: the model only ever emits
            // audio tokens (+ end). This cuts the LM-head matmul (~30% of a step) ~5×, shrinks the arena, and keeps
            // generation from emitting stray non-audio tokens. The range covers [end-of-text … end-of-vocab), which
            // contains the end token and the whole contiguous custom-token (audio) block.
            var fullVocab = _engine.Config.VocabSize;
            if (restrictToAudioVocab)
            {
                var audioBase = ResolveAudioTokenBase(Tokenizer);
                _outputStart = Math.Min(Tokenizer.EndOfTextTokenId, audioBase);
                _vocab = fullVocab - _outputStart;
            }
            else
            {
                _outputStart = 0;
                _vocab = fullVocab;
            }

            _model = TrainableLlamaModel.FromEngine(
                _engine, _opt.Rank, new Random(_opt.Seed), maxSeqLen: maxSeqLen + 8, loraOnLmHead: _opt.LoRAOnLmHead,
                outputStart: _outputStart, outputCount: restrictToAudioVocab ? _vocab : 0);

            // With checkpointing only one block's activations are live; without it, every block's are — so the arena
            // grows with the layer count and the FFN width.
            var cfg = _engine.Config;
            var blockActivations = useCheckpoint
                ? (long)(cfg.NLayers + 8) * maxSeqLen * cfg.DModel
                : (long)cfg.NLayers * maxSeqLen * ((8L * cfg.DModel) + (4L * cfg.DFF)) + (8L * maxSeqLen * cfg.DModel);
            var arena = (7L * maxSeqLen * _vocab) + blockActivations + 16_000_000L;
            _graph = new ComputationGraph((int)Math.Min(arena, int.MaxValue - 16));
        }

        /// <summary>The Orpheus tokenizer — pass it to <see cref="VoiceCloneDatasetBuilder"/> so prompts and audio
        /// tokens are encoded consistently with this trainer.</summary>
        public ITokenizer Tokenizer { get; }

        /// <summary>Llama end-of-text id (a reasonable default end-of-speech terminator for the dataset builder).</summary>
        public int EndOfTextTokenId => Tokenizer.EndOfTextTokenId;

        /// <summary>
        /// Fine-tunes on <paramref name="examples"/> for <see cref="QLoRAOptions.Epochs"/> passes. Each example is
        /// one step; the prompt tokens are masked so only the audio continuation is trained. Returns the per-step
        /// loss history; <paramref name="onStep"/> reports <c>(epoch, globalStep, loss)</c>.
        /// </summary>
        public IReadOnlyList<float> Train(IReadOnlyList<OrpheusTrainingExample> examples, Action<int, int, float>? onStep = null)
        {
            if (examples.Count == 0)
            {
                throw new ArgumentException("No training examples.", nameof(examples));
            }

            var trainable = MaterializeParams();
            using var optimizer = new Adam(trainable, _opt.LearningRate)
            {
                WeightDecay = 0f,
                Epsilon = _opt.AdamEpsilon,
            };

            var history = new List<float>(examples.Count * _opt.Epochs);
            var step = 0;
            for (var epoch = 0; epoch < _opt.Epochs; epoch++)
            {
                foreach (var ex in examples)
                {
                    if (ex.InputIds.Length > _maxSeqLen)
                    {
                        throw new OverfitRuntimeException(
                            $"Training example has {ex.InputIds.Length} tokens > maxSeqLen {_maxSeqLen}. Use shorter "
                            + "clips or raise maxSeqLen (costs RAM — the logits arena is ~7·T·vocab).");
                    }

                    var input = ex.InputIds[..^1];
                    var targets = ex.InputIds[1..];
                    // Mask predictions that fall inside the prompt: position i predicts InputIds[i+1]; train only
                    // where i+1 ≥ PromptLength (the audio-token continuation + end token). Trained targets are also
                    // shifted into the restricted output range (logits cover [_outputStart, _outputStart+_vocab)).
                    for (var i = 0; i < targets.Length; i++)
                    {
                        if (i < ex.PromptLength - 1)
                        {
                            targets[i] = IgnoreIndex;
                        }
                        else
                        {
                            targets[i] -= _outputStart;
                        }
                    }

                    _graph.Reset();
                    optimizer.ZeroGrad();
                    var logits = _model.Forward(_graph, input, useCheckpoint: _useCheckpoint);
                    var loss = TrainableLlamaModel.CrossEntropyLossAndSeed(logits, targets, _vocab, IgnoreIndex);
                    _graph.BackwardFromGrad(logits);
                    ClipGradNorm(trainable, _opt.GradientClipNorm);
                    optimizer.Step();

                    history.Add(loss);
                    onStep?.Invoke(epoch, step++, loss);
                }
            }
            return history;
        }

        /// <summary>Saves the trained LoRA adapter (small; the base GGUF is untouched).</summary>
        public void SaveAdapter(string path) => _model.SaveAdapter(path);

        /// <summary>Loads a previously-saved adapter (same base GGUF + options) for generation.</summary>
        public void LoadAdapter(string path) => _model.LoadAdapter(path);

        /// <summary>
        /// Generates a continuation of <paramref name="promptTokenIds"/> with the current adapter — used to
        /// synthesize in the cloned voice (feed the Orpheus prompt, get the audio-token stream back). Samples with
        /// the Orpheus defaults (temperature 0.6, top-p 0.9, repetition penalty 1.1); greedy decoding makes
        /// audio-token output drone/drag, so sampling is the default.
        /// </summary>
        public int[] Generate(
            int[] promptTokenIds, int maxNewTokens, int eosTokenId,
            float temperature = 0.6f, float topP = 0.9f, float repeatPenalty = 1.1f, int repeatWindow = 64, int seed = 0,
            int secondaryEosTokenId = -1)
            => _model.GenerateCachedSampled(
                promptTokenIds, maxNewTokens, eosTokenId, temperature, topP, repeatPenalty, repeatWindow, seed, secondaryEosTokenId);

        /// <summary>
        /// Bakes the trained LoRA + RMSNorm gains into the base and returns a fast, fully-quantized
        /// <see cref="CachedLlamaInferenceEngine"/> — the fine-tuned voice now decodes on the optimized
        /// zero-alloc/SIMD path (~2× the trainable graph). Keep this trainer alive for the engine's lifetime
        /// (it borrows the base embedding / LM-head handles).
        /// </summary>
        public CachedLlamaInferenceEngine BuildMergedEngine(bool mergeLora = true) => _model.BuildMergedEngine(_engine, mergeLora);

        /// <summary>Diagnostic: the trainable model's decode-path final hidden state for a prompt (to diff against a
        /// merged inference engine and localize divergence).</summary>
        public void DecodePromptHidden(int[] promptTokens, Span<float> hiddenOut) => _model.DecodePromptHidden(promptTokens, hiddenOut);

        /// <summary>The model config (dims) — for diagnostics/merge consumers.</summary>
        public GPT1Config Config => _engine.Config;

        // Real token id of <custom_token_0> — the bottom of the contiguous audio-token block.
        private static int ResolveAudioTokenBase(ITokenizer tokenizer)
        {
            Span<int> ids = stackalloc int[8];
            var n = tokenizer.Encode("<custom_token_0>", ids);
            if (n != 1)
            {
                throw new OverfitRuntimeException(
                    $"Expected '<custom_token_0>' to tokenize to one token, got {n} — not an Orpheus tokenizer.");
            }
            return ids[0];
        }

        private List<AutogradNode> MaterializeParams()
        {
            var list = new List<AutogradNode>();
            foreach (var p in _model.TrainableParameters())
            {
                list.Add(p);
            }
            return list;
        }

        private static void ClipGradNorm(List<AutogradNode> parameters, float maxNorm)
        {
            double sumSq = 0.0;
            foreach (var p in parameters)
            {
                var g = p.GradView.AsReadOnlySpan();
                for (var i = 0; i < g.Length; i++)
                {
                    sumSq += (double)g[i] * g[i];
                }
            }
            var norm = Math.Sqrt(sumSq);
            if (norm <= maxNorm || norm == 0.0)
            {
                return;
            }
            var scale = (float)(maxNorm / norm);
            foreach (var p in parameters)
            {
                var g = p.GradView.AsSpan();
                for (var i = 0; i < g.Length; i++)
                {
                    g[i] *= scale;
                }
            }
        }

        public void Dispose()
        {
            _graph.Dispose();
            _model.Dispose();
            _engine.Dispose();
        }
    }
}
