// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// A <see cref="FactAttribute"/> that RUNS on every local <c>dotnet test</c> but is SKIPPED on CI
    /// (GitHub Actions). For CPU-heavy training tests (QLoRA fine-tuning, TrainableLlama overfit/checkpointing)
    /// that finish in seconds on a dev box but take MINUTES on the 2–4 vCPU hosted runners — they are real
    /// correctness checks, so we keep them running locally and simply exclude them from CI. Unlike
    /// <see cref="LongFact"/> (skipped everywhere by default), this one only skips when a CI environment
    /// variable is present, so it needs no manual flip to run on the dev box.
    /// </summary>
    internal class LocalOnlyFact : FactAttribute
    {
        public LocalOnlyFact()
        {
            if (IsContinuousIntegration())
            {
                Skip = "Runs locally only — skipped on CI (CPU-heavy training; too slow on 2–4 vCPU hosted runners).";
            }
        }

        // GitHub Actions sets GITHUB_ACTIONS=true (and CI=true); most other CI systems set CI=true.
        internal static bool IsContinuousIntegration()
            => !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("GITHUB_ACTIONS"))
            || string.Equals(Environment.GetEnvironmentVariable("CI"), "true", StringComparison.OrdinalIgnoreCase);
    }
}
