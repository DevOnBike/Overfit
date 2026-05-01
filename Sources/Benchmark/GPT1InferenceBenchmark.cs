// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace Benchmarks
{
    /// <summary>
    /// GPT-1 inference benchmarks.
    ///
    /// Measures:
    ///   1. Single TransformerBlock forward [1, T, 768] — isolation of one block cost
    ///   2. Full GPT-1 forward [1, T, 768] × 12 blocks — end-to-end latency
    ///   3. GenerateLogits — single next-token prediction step
    ///
    /// Context lengths: 16, 64, 128 tokens.
    ///
    /// Machine: AMD Ryzen 9 9950X3D · .NET 10 · BenchmarkDotNet
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark --filter "*GPT1*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class GPT1InferenceBenchmark
    {
        // ── Params ────────────────────────────────────────────────────────────

        // SeqLen=128 requires >50MB arena (GPT-1 12 blocks × dFF=3072) — exceeds default capacity.
        // Use smaller context for benchmark; real training uses BackwardFromGrad path with full seqLen.
        [Params(16, 64)]
        public int SeqLen { get; set; }

        // ── State ─────────────────────────────────────────────────────────────

        private TransformerBlock _singleBlock = null!;
        private GPT1Model _gpt1Model = null!;
        private AutogradNode _blockInput = null!;
        private ComputationGraph _graph = null!;
        private int[] _tokenIds = null!;

        // ── GPT-1 config ─────────────────────────────────────────────────────

        private static readonly GPT1Config Gpt1Config = new()
        {
            VocabSize = 40478,
            ContextLength = 512,
            DModel = 768,
            NHeads = 12,
            NLayers = 12,
            DFF = 3072,
            TieWeights = true,
            PreLayerNorm = true,
        };

        // ── Setup ─────────────────────────────────────────────────────────────

        [GlobalSetup]
        public void Setup()
        {
            // Single block
            _singleBlock = new TransformerBlock(768, 12, 3072, causalMask: true);
            _singleBlock.Eval();

            // Full GPT-1
            _gpt1Model = new GPT1Model(Gpt1Config);
            _gpt1Model.Eval();

            // Token ids (random, within vocab)
            var rng = new Random(42);
            _tokenIds = new int[SeqLen];
            for (var i = 0; i < SeqLen; i++) _tokenIds[i] = rng.Next(0, Gpt1Config.VocabSize);

            // Pre-allocated block input [1, T, 768]
            var inputData = new float[1 * SeqLen * 768];
            for (var i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() - 0.5) * 0.02f;

            var storage = new TensorStorage<float>(inputData.Length, clearMemory: false);
            inputData.AsSpan().CopyTo(storage.AsSpan());
            _blockInput = new AutogradNode(storage, new TensorShape(1, SeqLen, 768), requiresGrad: false);

            _graph = new ComputationGraph();

            // Warmup
            for (var i = 0; i < 3; i++)
            {
                _graph.Reset();
                using var _ = _singleBlock.Forward(_graph, _blockInput);
                _gpt1Model.GenerateLogits(_tokenIds);
            }
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _singleBlock.Dispose();
            _gpt1Model.Dispose();
            _blockInput.Dispose();
            _graph.Dispose();
        }

        // ── Benchmarks ────────────────────────────────────────────────────────

        /// <summary>
        /// Single TransformerBlock forward [1, T, 768].
        /// Isolates: 1× MHA + 1× FFN + 2× LayerNorm.
        /// </summary>
        [Benchmark]
        public void GPT1_SingleBlock_Forward()
        {
            _singleBlock.InvalidateParameterCaches();
            _graph.Reset();
            using var output = _singleBlock.Forward(_graph, _blockInput);
        }

        /// <summary>
        /// Full GPT-1 next-token prediction: tokens → logits.
        /// 12 blocks × MHA + FFN + LN + embeddings + LM head.
        /// </summary>
        [Benchmark]
        public float[] GPT1_Full_GenerateLogits()
        {
            return _gpt1Model.GenerateLogits(_tokenIds);
        }
    }
}