// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// A <see cref="FactAttribute"/> for tests too slow for the default suite: real models loaded from
    /// <c>C:\qwen3b</c> and friends, training loops, profilers, PyTorch-parity diagnostics — anything over
    /// roughly ten seconds on the dev box. Skipped by default, so <c>dotnet test -c Release</c> stays fast and
    /// holds only correctness checks.
    ///
    /// <para><b>Set <c>OVERFIT_RUN_LONG=1</c> to run them.</b> Until 2026-08-06 there was no switch: the skip
    /// was unconditional and its own message told the reader to <i>remove the Skip property</i> — that is, to
    /// edit source in order to run a test. Measured that day: <b>358 of these against 1733 ordinary facts</b>,
    /// 277 under <c>LanguageModels</c>, which is the model loaders and the runtime. Nothing ran them on any
    /// schedule. <b>A test that never runs is worse than no test, because it looks like coverage</b> — and
    /// editing an attribute to run one has already gone wrong here, leaving a <c>[LongFact]</c> flipped to
    /// <c>[Fact]</c> until somebody noticed it by hand.
    ///
    /// <para>The environment variable is the pattern this repository already uses twice for a deliberate
    /// override of exactly this kind: <see cref="MeasurementExclusion"/> reads
    /// <c>OVERFIT_ALLOW_CONCURRENT_MEASUREMENT</c>, and <c>SmallModelFact</c> sets <c>Skip</c> conditionally
    /// on a fixture being present rather than unconditionally.</para>
    ///
    /// <para><b>The default is unchanged, deliberately.</b> Without the variable these skip exactly as before,
    /// so no existing run becomes slower by accident. Note also that a long run still takes the machine-wide
    /// measurement mutex through <see cref="MeasurementExclusion"/>: it refuses to start while a benchmark or
    /// an anomaly-guard measurement holds the box, which is correct — 358 model loads inside somebody's
    /// sampling window produces two wrong answers instead of one result.</para>
    /// </summary>
    internal class LongFact : FactAttribute
    {
        /// <summary>Set to <c>1</c> to run long tests instead of skipping them.</summary>
        internal const string RunVariable = "OVERFIT_RUN_LONG";

        public LongFact()
        {
            // Polarity matters more here than anything else in the file: skipping is the DEFAULT and only an
            // explicit "1" lifts it. Inverting this would silently pull 358 model-loading tests into every
            // `dotnet test` and turn a fast suite into an hours-long one — a failure that looks like the
            // suite simply got slow.
            if (Environment.GetEnvironmentVariable(RunVariable) == "1")
            {
                return;
            }

            Skip = "Long-running: skipped by default. Set OVERFIT_RUN_LONG=1 to run these — they load real "
                + "models, take minutes, and hold the machine measurement mutex while they do.";
        }
    }
}
