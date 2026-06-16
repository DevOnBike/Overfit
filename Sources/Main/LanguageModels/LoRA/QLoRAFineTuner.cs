// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.IO;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Optimizers;

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    /// <summary>
    /// Turnkey QLoRA fine-tuning for an already-quantized Qwen/Llama GGUF — pure .NET, CPU, no GPU/Python.
    /// Loads the 4-bit base + tokenizer, fine-tunes a tiny trainable LoRA adapter on your text (the frozen
    /// base is never expanded to F32 or rewritten), saves/loads the adapter as a small portable file, and
    /// answers prompts so you can see the new knowledge. Wraps <see cref="TrainableLlamaModel"/> with the
    /// validated known-good config baked in (see <see cref="QLoRAOptions"/>).
    ///
    /// <code>
    ///   using var ft = new QLoRAFineTuner(@"C:\qwen3b\qwen.q4km.gguf");
    ///   ft.FineTuneOnFile("notes.txt");
    ///   ft.SaveAdapter("notes.lora");
    ///   Console.WriteLine(ft.Ask("Summarize the notes:"));
    /// </code>
    ///
    /// Expects the tokenizer (<c>tokenizer.json</c>) in the GGUF's directory. Scope: batch = 1 (one
    /// sequence per step), Qwen tokenizer. Training cost ≈ 0.084 s/token on a typical CPU.
    /// </summary>
    public sealed class QLoRAFineTuner : IDisposable
    {
        private readonly QLoRAOptions _opt;
        private readonly CachedLlamaInferenceEngine _engine;
        private readonly QwenTokenizer _tokenizer;
        private readonly TrainableLlamaModel _model;
        private readonly ComputationGraph _graph;
        private readonly int _vocab;

        /// <summary>Loads the GGUF base + its tokenizer and builds a fresh trainable adapter.</summary>
        /// <param name="ggufPath">Path to the quantized <c>.gguf</c> (its directory must also hold <c>tokenizer.json</c>).</param>
        /// <param name="options">Fine-tuning settings, or null for the validated defaults.</param>
        public QLoRAFineTuner(string ggufPath, QLoRAOptions? options = null)
        {
            _opt = options ?? new QLoRAOptions();
            _engine = CachedLlamaInferenceEngine.LoadGguf(ggufPath);
            var dir = Path.GetDirectoryName(Path.GetFullPath(ggufPath))
                ?? throw new ArgumentException($"Could not resolve directory of '{ggufPath}'.", nameof(ggufPath));
            _tokenizer = QwenTokenizer.Load(dir);
            _vocab = _engine.Config.VocabSize;

            _model = TrainableLlamaModel.FromEngine(
                _engine, _opt.Rank, new Random(_opt.Seed), maxSeqLen: _opt.ChunkLength + 8, loraOnLmHead: _opt.LoRAOnLmHead);

            // Main arena: the [T, vocab] logits region (base + LM-head LoRA + add, each with grad ≈ 6×) plus
            // the per-layer checkpointed outputs. Generous, sized once for the chosen chunk length.
            var T = _opt.ChunkLength;
            var arena = 7L * T * _vocab
                + (long)(_engine.Config.NLayers + 8) * T * _engine.Config.DModel
                + 16_000_000L;
            _graph = new ComputationGraph((int)Math.Min(arena, int.MaxValue - 16));
        }

        /// <summary>Transformer layer count of the loaded model.</summary>
        public int LayerCount => _model.LayerCount;

        /// <summary>Fine-tune on the contents of a text file. See <see cref="FineTune"/>.</summary>
        public IReadOnlyList<float> FineTuneOnFile(string textPath, Action<int, int, float>? onStep = null)
            => FineTune(File.ReadAllText(textPath), onStep);

        /// <summary>
        /// Fine-tune the adapter on <paramref name="text"/>. The text is tokenized and split into
        /// non-overlapping windows of <see cref="QLoRAOptions.ChunkLength"/>; each window is a next-token
        /// training step, repeated for <see cref="QLoRAOptions.Epochs"/> passes. The frozen 4-bit base is
        /// never modified. Returns the per-step loss history; <paramref name="onStep"/> is invoked as
        /// <c>(epoch, globalStep, loss)</c> for progress reporting.
        /// </summary>
        public IReadOnlyList<float> FineTune(string text, Action<int, int, float>? onStep = null)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                throw new ArgumentException("Training text is empty.", nameof(text));
            }

            var ids = _tokenizer.Encode(text);
            var chunks = Chunk(ids, _opt.ChunkLength);
            if (chunks.Count == 0)
            {
                throw new ArgumentException("Text too short to form a single training sequence.", nameof(text));
            }

            var trainable = MaterializeParams();
            using var optimizer = new Adam(trainable, _opt.LearningRate)
            {
                WeightDecay = 0f,
                Epsilon = _opt.AdamEpsilon,
            };

            var history = new List<float>(chunks.Count * _opt.Epochs);
            var step = 0;
            for (var epoch = 0; epoch < _opt.Epochs; epoch++)
            {
                foreach (var chunk in chunks)
                {
                    var input = chunk[..^1];
                    var target = chunk[1..];

                    _graph.Reset();
                    optimizer.ZeroGrad();
                    var logits = _model.Forward(_graph, input, useCheckpoint: true);
                    var loss = TrainableLlamaModel.CrossEntropyLossAndSeed(logits, target, _vocab);
                    _graph.BackwardFromGrad(logits);
                    ClipGradNorm(trainable, _opt.GradientClipNorm);
                    optimizer.Step();

                    history.Add(loss);
                    onStep?.Invoke(epoch, step++, loss);
                }
            }
            return history;
        }

        /// <summary>Greedily generate a continuation of <paramref name="prompt"/> (decoded to text).
        /// Reflects the current adapter state — call after <see cref="FineTune"/> or <see cref="LoadAdapter"/>.</summary>
        public string Ask(string prompt, int maxNewTokens = 32)
        {
            var promptTokens = _tokenizer.Encode(prompt);
            var produced = _model.Generate(_graph, promptTokens, maxNewTokens, QwenTokenizer.EndOfText);
            return _tokenizer.Decode(produced);
        }

        /// <summary>Saves only the trained adapter (a small file; the base GGUF is untouched).</summary>
        public void SaveAdapter(string path) => _model.SaveAdapter(path);

        /// <summary>Loads an adapter saved by <see cref="SaveAdapter"/> into this model (same GGUF + options).</summary>
        public void LoadAdapter(string path) => _model.LoadAdapter(path);

        // ── helpers ──

        private static List<int[]> Chunk(ReadOnlySpan<int> ids, int chunkLength)
        {
            // Non-overlapping windows of (chunkLength + 1) tokens → input[..^1] + target[1..]. A trailing
            // window shorter than 2 tokens is dropped (can't form an input/target pair).
            var window = chunkLength + 1;
            var chunks = new List<int[]>();
            for (var start = 0; start + 2 <= ids.Length; start += chunkLength)
            {
                var len = Math.Min(window, ids.Length - start);
                if (len < 2) { break; }
                chunks.Add(ids[start..(start + len)].ToArray());
            }
            return chunks;
        }

        private List<AutogradNode> MaterializeParams()
        {
            var list = new List<AutogradNode>();
            foreach (var p in _model.TrainableParameters()) { list.Add(p); }
            return list;
        }

        private static void ClipGradNorm(List<AutogradNode> parameters, float maxNorm)
        {
            double sq = 0;
            foreach (var p in parameters)
            {
                var g = p.GradView.AsReadOnlySpan();
                for (var i = 0; i < g.Length; i++) { sq += (double)g[i] * g[i]; }
            }
            var norm = Math.Sqrt(sq);
            if (norm <= maxNorm) { return; }
            var scale = (float)(maxNorm / (norm + 1e-6));
            foreach (var p in parameters)
            {
                var g = p.GradView.AsSpan();
                for (var i = 0; i < g.Length; i++) { g[i] *= scale; }
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
