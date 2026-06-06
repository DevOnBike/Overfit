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
        private readonly int _vocab;
        private readonly int _maxSeqLen;

        public VoiceCloneTrainer(string orpheusGgufPath, int maxSeqLen = 768, QLoRAOptions? options = null)
        {
            _opt = options ?? new QLoRAOptions();
            _maxSeqLen = maxSeqLen;
            _engine = CachedLlamaInferenceEngine.LoadGguf(orpheusGgufPath);
            _vocab = _engine.Config.VocabSize;
            Tokenizer = new GgufEmbeddedTokenizer(GgufTokenizer.Load(orpheusGgufPath));

            _model = TrainableLlamaModel.FromEngine(
                _engine, _opt.Rank, new Random(_opt.Seed), maxSeqLen: maxSeqLen + 8, loraOnLmHead: _opt.LoRAOnLmHead);

            var arena = 7L * maxSeqLen * _vocab
                + (long)(_engine.Config.NLayers + 8) * maxSeqLen * _engine.Config.DModel
                + 16_000_000L;
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
                    // where i+1 ≥ PromptLength (the audio-token continuation + end token).
                    for (var i = 0; i < ex.PromptLength - 1 && i < targets.Length; i++)
                    {
                        targets[i] = IgnoreIndex;
                    }

                    _graph.Reset();
                    optimizer.ZeroGrad();
                    var logits = _model.Forward(_graph, input, useCheckpoint: true);
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
        /// Greedily generates a continuation of <paramref name="promptTokenIds"/> with the current adapter — used to
        /// synthesize in the cloned voice: feed the Orpheus prompt, get the audio-token stream back.
        /// </summary>
        public int[] Generate(int[] promptTokenIds, int maxNewTokens, int eosTokenId)
            => _model.GenerateCached(promptTokenIds, maxNewTokens, eosTokenId);

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
