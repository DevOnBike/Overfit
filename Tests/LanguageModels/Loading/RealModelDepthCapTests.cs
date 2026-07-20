// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Validates the OVERFIT022 depth caps against REAL model files rather than against an assumption.
    ///
    /// <para>The caps (GGUF array nesting = 8, tokenizer.json pre-tokenizers = 16, JSON schema = 32) were
    /// introduced with the claim "real models nest at most one level". That claim was asserted, not measured —
    /// and the fast suite cannot catch a wrong cap because every real-model test is <c>[Fact]</c>, i.e.
    /// skipped by default. A cap set too low would break every model load in production while the suite stayed
    /// green. This test measures the actual nesting in the model fixtures so the margin is a number, not a hope.</para>
    /// </summary>
    public sealed class RealModelDepthCapTests
    {
        private static readonly string[] Candidates =
        [
            @"C:\qwen3b\qwen0.5b.q4km.gguf",
            @"C:\qwen3b\qwen.q4km.gguf",
            @"C:\gemma\gemma-2-2b-it-Q4_K_M.gguf",
        ];

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

        [Fact]
        public void RealGgufMetadata_NestsFarBelowTheCap()
        {
            var checkedAny = false;

            foreach (var path in Candidates)
            {
                if (!File.Exists(path))
                {
                    continue;
                }

                checkedAny = true;

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

                // The cap is 8. Anything at or above it means the cap is wrong, not the file.
                Assert.True(
                    deepest < 8,
                    $"{Path.GetFileName(path)}: deepest metadata nesting is {deepest} (key '{deepestKey}') — "
                    + "at or above the MaxValueNestingDepth cap of 8, so the cap would reject a real model.");
            }

            Assert.True(checkedAny, "No model fixture found — this test proves nothing unless a file was read.");
        }
    }
}
