// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// A <see cref="LongFact"/> that additionally skips when the trained anomaly base
    /// (<c>k8s_anomaly_production.bin</c>) is not on the box.
    ///
    /// <para><b>Why an attribute rather than a check inside the test.</b> Until 2026-08-07 the test read
    /// the fixture path, found nothing, wrote a line to the output and <c>return</c>ed — which is a
    /// <b>pass</b>. It had never run, so nobody had seen it: the one test whose entire purpose is to
    /// confirm the LoRA target recommendation <i>on the real trained artifact</i> was reporting success
    /// having loaded nothing. A green result from a run that did no work is the most misleading outcome
    /// available, because it answers "was this checked?" with "yes".</para>
    ///
    /// <para>xUnit 2.9.3 has no dynamic skip — <c>Assert.Skip</c> arrived in v3, and trying it here is a
    /// compile error (measured, not assumed). The repository's own answer to this predates the problem:
    /// <see cref="SmallModelFact"/> and <see cref="Gpt2ModelFact"/> both set <c>Skip</c> from a fixture
    /// check at discovery time. This is the same shape, and it keeps the <c>OVERFIT_RUN_LONG</c> gate by
    /// deriving from <see cref="LongFact"/> rather than from <c>FactAttribute</c> directly.</para>
    ///
    /// <para>The skip message names <b>every directory that was searched</b>. "Not found" without a search
    /// path tells the reader nothing they can act on.</para>
    /// </summary>
    internal sealed class ProductionAnomalyBaseFact : LongFact
    {
        /// <summary>The checkpoint this fixture-dependent group needs.</summary>
        internal const string FileName = "k8s_anomaly_production.bin";

        public ProductionAnomalyBaseFact(string runtime = null)
            : base(runtime)
        {
            // Already skipped as a long test — leave that message alone, it is the more general reason.
            if (Skip is not null)
            {
                return;
            }

            if (Resolve() is not null)
            {
                return;
            }

            Skip = $"{FileName} not found. Looked in OVERFIT_MODEL_DIR, test_fixtures/ and D:\\. Set "
                + "OVERFIT_MODEL_DIR to the directory holding it, or train one with the anomaly training "
                + "demo.";
        }

        /// <summary>The checkpoint's path, or <see langword="null"/> if it is on none of the three routes.</summary>
        internal static string Resolve()
        {
            var configured = Environment.GetEnvironmentVariable("OVERFIT_MODEL_DIR");

            string[] candidates =
            [
                configured is null ? null : Path.Combine(configured, FileName),
                Path.Combine("test_fixtures", FileName),
                @"D:\" + FileName,
            ];

            foreach (var candidate in candidates)
            {
                if (candidate is not null && File.Exists(candidate))
                {
                    return candidate;
                }
            }

            return null;
        }
    }
}
