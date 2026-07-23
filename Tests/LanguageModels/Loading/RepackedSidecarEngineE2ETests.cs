// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Increment 2b end-to-end gate: the loader auto-discovers a <c>&lt;model&gt;.repack</c> sidecar and mmaps its
    /// <c>block_q4_Kx8</c> slices into the repackable Q4_K weights, so <c>EnsureRepacked</c> hands them out with
    /// zero heap copy. Because those bytes are byte-identical to the runtime repack, a full generation with the
    /// tiled-prefill path ON must be <b>bit-exact</b> whether the sidecar is present or not — proving the
    /// consumed sidecar changes nothing about the computation. [LongFact] — needs C:\qwen3b\qwen.q4km.gguf.
    /// </summary>
    public sealed class RepackedSidecarEngineE2ETests
    {
        private const string Gguf = @"C:\qwen3b\qwen.q4km.gguf";
        private const int Context = 1024;
        private const int GenTokens = 16;

        private readonly ITestOutputHelper _out;

        public RepackedSidecarEngineE2ETests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Sidecar_ConsumedByLoader_BitExactToRuntimeRepack()
        {
            if (!File.Exists(Gguf))
            {
                _out.WriteLine("missing gguf — skipping");
                return;
            }

            var sidecar = Gguf + ".repack";
            var preexisting = File.Exists(sidecar);
            var prevToggle = BatchedQuantProjection.UseTiledPrefillQ4K;
            try
            {
                BatchedQuantProjection.UseTiledPrefillQ4K = true; // exercise the EnsureRepacked path

                var tok = GgufTokenizer.Load(Gguf);
                var prompt = tok.Encode(
                    "The history of computing is a long and winding road that begins with mechanical "
                    + "calculators and arrives at modern processors. In a few sentences, summarise it:");

                // Sidecar PRESENT → repacked weights come from the mmap'd file.
                var count = RepackedWeightsFile.BuildFromGguf(Gguf, sidecar);
                Assert.True(count > 0, "expected repackable Q4_K tensors");
                _out.WriteLine($"sidecar tensors: {count}");
                var withSidecar = Generate(prompt);

                // Sidecar ABSENT → repacked weights are built at runtime on the heap.
                File.Delete(sidecar);
                var withoutSidecar = Generate(prompt);

                _out.WriteLine($"with:    {string.Join(",", withSidecar)}");
                _out.WriteLine($"without: {string.Join(",", withoutSidecar)}");
                Assert.Equal(withoutSidecar, withSidecar); // byte-identical repack ⇒ identical generation
            }
            finally
            {
                BatchedQuantProjection.UseTiledPrefillQ4K = prevToggle;
                if (!preexisting && File.Exists(sidecar))
                {
                    File.Delete(sidecar);
                }
            }
        }

        private static List<int> Generate(int[] prompt)
        {
            using var engine = CachedLlamaInferenceEngine.LoadGguf(Gguf);
            var sampling = SamplingOptions.Greedy;
            using var session = engine.CreateSession(Context);
            session.Reset(prompt);
            var ids = new List<int>(GenTokens);
            for (var i = 0; i < GenTokens && !session.IsFull; i++)
            {
                ids.Add(session.GenerateNextToken(in sampling));
            }
            return ids;
        }
    }
}
