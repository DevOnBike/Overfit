// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// A/B of the Q4_K batched-prefill kernels that <see cref="BatchedQuantProjection"/> can actually switch,
    /// on ONE loaded model in ONE process. Prefill is 99.1% of time-to-first-token
    /// (<see cref="PrefillProfileTests"/>), so this is the path worth measuring.
    ///
    /// <para><b>Method — two traps this measurement already fell into.</b> (1) A cross-process before/after
    /// showed a +5% "regression" while the untouched decode path moved +32%: the box had drifted, not the
    /// code. So configurations are <b>interleaved</b> run-by-run and every sample also times a first-token
    /// decode as a <b>canary</b> — decode uses a different kernel, so a material canary drift invalidates the
    /// prefill numbers. (2) An attempt to A/B <c>UseTiledPrefillQ4K</c> measured nothing at all, because a
    /// <c>*.gguf.repack</c> sidecar sets <c>IsPrepacked</c>, which short-circuits that flag in the tiled
    /// gate — both arms silently ran the identical mix.</para>
    ///
    /// <para><b>Recorded negative — do not re-try without new evidence.</b> The tiled GEMM is barred from
    /// biased projections by the <c>bias.IsEmpty</c> term in its gate, and a path census showed that excludes
    /// 88% of Q4_K prefill dispatches (only 540 of 4644 were bias-free; AVX2/FMA and <c>CanRepack</c> held
    /// for 100%). Adding an optional bias to <c>GemmTiled</c> to lift that restriction was implemented,
    /// pinned bit-identical, and measured at <b>0.999× — an exact tie</b> (raw samples fully interleaved), so
    /// it was reverted. The reason it ties: <c>ProjectBatchedWeightStationary</c> already decodes each
    /// super-block once and reuses it across the row tile, i.e. it amortises exactly what the tiling
    /// amortises. The "~3×" in the kernel docs is against <c>ProjectBatched</c> (re-decode per row), not
    /// against weight-stationary — which is the ratio this test measures.</para>
    /// </summary>
    public sealed class PrefillPathAbTests
    {
        private readonly ITestOutputHelper _out;

        public PrefillPathAbTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Prefill_WeightStationaryVsReDecode()
        {
            var path = TestModelPaths.Qwen3B.Q4KmGgufPath;
            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path}");
                return;
            }

            var original = BatchedQuantProjection.UseWeightStationaryQ4K;

            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var tok = GgufTokenizer.Load(path);

            var paragraph = string.Join(" ",
                Enumerable.Repeat(
                    "The history of computing began with mechanical calculators and evolved through vacuum tubes, "
                    + "transistors, integrated circuits and finally the microprocessor era.", 24));
            var ids = tok.Encode(paragraph);

            const int Runs = 7;
            var onPrefill = new double[Runs];
            var offPrefill = new double[Runs];
            var onDecode = new double[Runs];
            var offDecode = new double[Runs];
            var onToken = 0;
            var offToken = 0;

            try
            {
                Warm(engine, ids, weightStationary: true);
                Warm(engine, ids, weightStationary: false);

                // Interleaved so any monotonic drift (thermal, background load) hits both arms equally.
                for (var r = 0; r < Runs; r++)
                {
                    (onPrefill[r], onDecode[r], onToken) = Sample(engine, ids, weightStationary: true);
                    (offPrefill[r], offDecode[r], offToken) = Sample(engine, ids, weightStationary: false);
                }

                var onP = Median(onPrefill);
                var offP = Median(offPrefill);
                var onD = Median(onDecode);
                var offD = Median(offDecode);

                _out.WriteLine($"=== Q4_K prefill kernels ({ids.Length} tokens, median of {Runs}, interleaved) ===");
                _out.WriteLine($"  weight-stationary : {onP,9:F1} ms   {ids.Length / (onP / 1000.0),7:F0} tok/s   (default)");
                _out.WriteLine($"  re-decode-per-row : {offP,9:F1} ms   {ids.Length / (offP / 1000.0),7:F0} tok/s");
                _out.WriteLine($"  speedup           : {offP / onP,9:F3}x");
                _out.WriteLine(string.Empty);
                _out.WriteLine($"  CANARY decode     : {onD,6:F1} ms vs {offD,6:F1} ms   drift {100.0 * (onD - offD) / offD,5:F1}%");
                _out.WriteLine("    (decode shares no kernel with these — material drift invalidates the comparison)");
                _out.WriteLine($"  first token id    : {onToken} / {offToken}");
                _out.WriteLine($"  repack sidecar    : {File.Exists(path + ".repack")} "
                    + "(when true, bias-free projections take the tiled GEMM regardless of OVERFIT_TILED_PREFILL)");

                // Swapping kernels must not change what the model produces.
                Assert.Equal(onToken, offToken);
            }
            finally
            {
                BatchedQuantProjection.UseWeightStationaryQ4K = original;
            }
        }

        private static void Warm(CachedLlamaInferenceEngine engine, int[] ids, bool weightStationary)
        {
            BatchedQuantProjection.UseWeightStationaryQ4K = weightStationary;
            var sampling = SamplingOptions.Greedy;
            using var warm = engine.CreateSession(1024);
            warm.Reset(ids);
            warm.GenerateNextToken(in sampling);
        }

        private static (double Prefill, double Decode, int Token) Sample(
            CachedLlamaInferenceEngine engine, int[] ids, bool weightStationary)
        {
            BatchedQuantProjection.UseWeightStationaryQ4K = weightStationary;
            var sampling = SamplingOptions.Greedy;

            using var session = engine.CreateSession(1024);

            var sw = Stopwatch.StartNew();
            session.Reset(ids);
            sw.Stop();
            var prefill = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            var token = session.GenerateNextToken(in sampling);
            sw.Stop();

            return (prefill, sw.Elapsed.TotalMilliseconds, token);
        }

        private static double Median(double[] values)
        {
            var copy = (double[])values.Clone();
            Array.Sort(copy);
            return copy[copy.Length / 2];
        }
    }
}
