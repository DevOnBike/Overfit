// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Idle-burn gate for the decode spin-pool (spin-then-park, 2026-06-11): after a decode finishes,
    /// the pool must PARK — a serving container at rest must not spin cores (field report: 100% CPU at
    /// idle on a 16-core laptop with the pre-park pool). Decodes a few tokens to wake the pool, then
    /// sleeps and asserts the process burns &lt; 1 effective core during the idle window (parked
    /// pool ≈ 0; the old pure-spin pool burned ~10). [LongFact] — needs C:\qwen3-06b.
    ///
    /// <para><b>The instrument is process-wide, so the test needs two things to be honest about it
    /// (<c>XC-54</c>, 2026-08-15).</b> <see cref="Process.TotalProcessorTime"/> counts every thread in the
    /// process, and xUnit runs collections in parallel: in an area run this assertion measured
    /// <b>14.57 effective cores on a 32-logical-core box</b> — real CPU, none of it the pool's — while
    /// passing in isolation. So (1) the class sits in
    /// <see cref="ExclusiveProcessMeasurementCollection"/>, which serialises it against the whole suite,
    /// and (2) it takes a <b>canary</b>: the same measurement over an equal window <b>before the pool is
    /// ever woken</b>.</para>
    ///
    /// <para><b>The verdict is three-valued, which is the point of the canary.</b> Quiet baseline + quiet
    /// idle window is a pass; quiet baseline + busy idle window is a failure that means what the test
    /// claims; a <b>busy baseline is neither</b> — the instrument cannot separate the pool from whatever
    /// else is burning, so the test skips with the measured baseline in the message rather than picking
    /// one of the two answers. A test that cannot tell its subject from its environment must say so.</para>
    ///
    /// <para><b>Known limitation, deliberately recorded rather than fixed.</b> Load from OTHER processes
    /// does not enter this counter, but it does bias the result <b>downwards</b>: a genuinely spinning pool
    /// starved of cores accumulates less CPU per wall second, so a heavily loaded box can make this test
    /// falsely quiet. The bias is in the safe direction for a false RED and the unsafe one for a false
    /// GREEN. Sampling the pool's own worker threads instead of the process would remove both the canary
    /// and this caveat; it was judged the more expensive fix in <c>XC-54</c> and is not done here.</para>
    /// </summary>
    [Collection(ExclusiveProcessMeasurementCollection.Name)]
    public sealed class DecodePoolIdleBurnTests
    {
        private const string Path = @"C:\qwen3-06b\Qwen3-0.6B-Q4_K_M.gguf";

        /// <summary>
        /// Length of BOTH measurement windows. They must be equal: the canary is only a control for the
        /// idle window if it is the same measurement over the same span.
        /// </summary>
        private const int WindowMilliseconds = 3000;

        /// <summary>
        /// The gate. A parked pool reads ≈ 0; the pre-park pure-spin pool read ~10-15 on this box.
        /// </summary>
        private const double IdleBurnLimitCores = 1.0;

        /// <summary>
        /// Above this, the baseline window is judged contaminated and the test reports that it could not
        /// measure. Chosen so that a passing run still leaves at least three quarters of a core of
        /// headroom between ambient noise and <see cref="IdleBurnLimitCores"/> — i.e. so a pass cannot be
        /// bought by ambient CPU that happens to sit just under the gate. An uncontended run measures
        /// ~0.0X here, so this is not a tight budget; it is a contamination detector.
        /// </summary>
        private const double BaselineQuietCores = 0.25;

        private readonly ITestOutputHelper _out;
        public DecodePoolIdleBurnTests(ITestOutputHelper output) => _out = output;

        [ModelFact(Path, "10s")]
        public void Pool_Parks_WhenIdle()
        {
            // CANARY FIRST — before anything is loaded, so nothing this test owns can be spinning yet.
            // Whatever this reads is the floor the real measurement is standing on.
            var baseline = MeasureEffectiveCores(out var baselineWindowSeconds);
            _out.WriteLine($"baseline window {baselineWindowSeconds:F1}s: {baseline:F2} effective cores (pool never woken)");

            Assert.SkipWhen(
                baseline >= BaselineQuietCores,
                $"cannot measure: the process was already burning {baseline:F2} effective cores before the "
                + $"pool was woken (limit {BaselineQuietCores:F2}). This is NOT a pass and NOT a failure — "
                + "process CPU cannot separate the decode pool from whatever else is running here.");

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);
            using var session = engine.CreateSession(128);
            session.Reset(tok.Encode("Hello"));
            var sampling = SamplingOptions.Greedy;
            for (var i = 0; i < 8; i++)
            {
                session.GenerateNextToken(in sampling);
            }   // pool is hot now

            var effectiveCores = MeasureEffectiveCores(out var idleWindowSeconds);

            _out.WriteLine($"idle window {idleWindowSeconds:F1}s: {effectiveCores:F2} effective cores (pool woken, then idle)");
            Assert.True(effectiveCores < IdleBurnLimitCores,
                $"decode pool is burning CPU at idle: {effectiveCores:F2} effective cores against a baseline "
                + $"of {baseline:F2} before it was woken (expected ~0 after spin-then-park)");
        }

        /// <summary>
        /// Process CPU consumed per wall second over one idle window: the whole-process rate, in cores.
        /// </summary>
        private static double MeasureEffectiveCores(out double windowSeconds)
        {
            using var process = Process.GetCurrentProcess();
            process.Refresh();
            var cpuBefore = process.TotalProcessorTime;
            var sw = Stopwatch.StartNew();
            Thread.Sleep(WindowMilliseconds);
            sw.Stop();
            process.Refresh();
            windowSeconds = sw.Elapsed.TotalSeconds;

            return (process.TotalProcessorTime - cpuBefore).TotalSeconds / windowSeconds;
        }
    }
}
