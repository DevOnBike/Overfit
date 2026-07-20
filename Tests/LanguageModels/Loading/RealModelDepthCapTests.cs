// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Validates the OVERFIT022 depth caps against REAL model files rather than against an assumption.
    ///
    /// <para>The caps (GGUF array nesting = 8, tokenizer.json pre-tokenizers = 16, JSON schema = 32) were
    /// introduced with the claim "real models nest at most one level". That claim was asserted, not measured.
    /// A cap set too low would reject every model in production, so the margin needs to be a number: measured
    /// on the fixtures, the deepest real GGUF metadata nesting is <b>1</b> against a cap of 8.</para>
    ///
    /// <para>Marked <see cref="SmallModelFact"/>, not <c>[Fact]</c>: it needs a model fixture on disk and must
    /// SKIP where there is none (CI) rather than fail. An earlier revision of this file was committed as a
    /// plain <c>[Fact]</c> and broke the CI run for exactly that reason — the assert below is designed to fail
    /// when nothing was checked, which is right for a dev box and wrong for a runner with no fixtures.</para>
    /// </summary>
    public sealed class RealModelDepthCapTests
    {
        /// <summary>Deepest array nesting in a metadata value tree (a scalar is depth 0).</summary>
        private static int MeasureDepth(object value)
        {
            if (value is not object[] array)
            {
                return 0;
            }

            var deepest = 0;
            foreach (var element in array)
            {
                var d = MeasureDepth(element);
                if (d > deepest)
                {
                    deepest = d;
                }
            }
            return deepest + 1;
        }

        [SmallModelFact]
        public void RealGgufMetadata_NestsFarBelowTheCap()
        {
            // Resolved through TestModelPaths so OVERFIT_QWEN3B_DIR works and the path is not hard-coded to
            // a Windows drive — the same fixture check SmallModelFact uses to decide whether to skip.
            var path = TestModelPaths.Qwen05B.RequireQ4KmGgufPath();

            using var reader = new GgufReader(path);

            var deepest = 0;
            var deepestKey = string.Empty;
            foreach (var kv in reader.Metadata)
            {
                var d = MeasureDepth(kv.Value);
                if (d > deepest)
                {
                    deepest = d;
                    deepestKey = kv.Key;
                }
            }

            // Measured margin, not a hope: real nesting is 1, the cap is 8.
            Assert.True(
                deepest < 8,
                $"{Path.GetFileName(path)}: deepest metadata nesting is {deepest} (key '{deepestKey}') — "
                + "at or above the MaxValueNestingDepth cap of 8, so the cap would reject a real model.");

            // A run that parsed no nested value would pass the assert above while proving nothing.
            Assert.True(deepest > 0, "No nested metadata value found — this fixture does not exercise the cap.");
        }
    }
}
