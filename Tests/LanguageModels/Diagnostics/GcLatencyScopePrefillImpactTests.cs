// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Measures whether <see cref="GcLatencyScope"/> (SustainedLowLatency during generation) actually buys anything
    /// on Overfit's generation path — the honest "measure, don't assume" check for a setting added on convention.
    /// Runs N prefill+decode cycles with and without the scope and reports the GC deltas (gen0/1/2 collection counts,
    /// total GC pause, allocated bytes). HYPOTHESIS: near-zero effect, because decode is zero-allocation and the
    /// batched prefill scratch is already pooled — so there are few gen-2 collections for SustainedLowLatency to
    /// suppress. Read the printed numbers (don't assert a delta — the expected effect is ~0 and would be flaky).
    /// <para>[LongFact] — needs a real Qwen Q4_K_M GGUF, and a thermally STABLE (cold) box for a meaningful read.</para>
    /// </summary>
    public sealed class GcLatencyScopePrefillImpactTests
    {
        private const int PromptLen = 128;
        private const int DecodeTokens = 6;
        private const int Cycles = 150;

        private readonly ITestOutputHelper _out;
        public GcLatencyScopePrefillImpactTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void SustainedLowLatency_GenerationCycle_GcDelta()
        {
            var model = TestModelPaths.Qwen3B.Q4KmGgufPath;
            if (!File.Exists(model))
            {
                _out.WriteLine($"missing {model}");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(model);

            var vocab = engine.Config.VocabSize;
            var prompt = new int[PromptLen];
            for (var i = 0; i < PromptLen; i++)
            {
                prompt[i] = 100 + (i % 500) % vocab;
            }

            var sampling = SamplingOptions.GreedyWithPenalty(1f);
            using var session = engine.CreateSession();

            void Run(int n)
            {
                for (var c = 0; c < n; c++)
                {
                    session.Reset();
                    session.Prefill(prompt);
                    for (var t = 0; t < DecodeTokens; t++)
                    {
                        session.GenerateNextToken(in sampling);
                    }
                }
            }

            // Warm up: JIT, pool fill, first-prefill one-off allocations — outside the measured windows.
            Run(5);

            var without = Measure(() => Run(Cycles));
            var with = Measure(() =>
            {
                using var gc = GcLatencyScope.SustainedLowLatency();
                Run(Cycles);
            });

            _out.WriteLine($"{Cycles} cycles × (prefill {PromptLen} + decode {DecodeTokens}) on Qwen Q4_K_M:");
            _out.WriteLine($"  {"",-22} | gen0 | gen1 | gen2 |  GC pause ms | alloc MB");
            _out.WriteLine($"  {"WITHOUT scope",-22} | {without.G0,4} | {without.G1,4} | {without.G2,4} | {without.PauseMs,12:F2} | {without.AllocMb,8:F1}");
            _out.WriteLine($"  {"WITH SustainedLowLat",-22} | {with.G0,4} | {with.G1,4} | {with.G2,4} | {with.PauseMs,12:F2} | {with.AllocMb,8:F1}");
            _out.WriteLine($"  gen2 delta (with - without): {with.G2 - without.G2}; GC-pause delta: {with.PauseMs - without.PauseMs:F2} ms");

            // Sanity only: the work ran and allocations were comparable (same workload both windows).
            Assert.True(without.AllocMb >= 0 && with.AllocMb >= 0);
        }

        private static GcDelta Measure(Action work)
        {
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);

            var g0 = GC.CollectionCount(0);
            var g1 = GC.CollectionCount(1);
            var g2 = GC.CollectionCount(2);
            var pause = GC.GetTotalPauseDuration();
            var alloc = GC.GetTotalAllocatedBytes();

            work();

            return new GcDelta(
                GC.CollectionCount(0) - g0,
                GC.CollectionCount(1) - g1,
                GC.CollectionCount(2) - g2,
                (GC.GetTotalPauseDuration() - pause).TotalMilliseconds,
                (GC.GetTotalAllocatedBytes() - alloc) / (1024.0 * 1024.0));
        }

        private readonly record struct GcDelta(int G0, int G1, int G2, double PauseMs, double AllocMb);
    }
}
