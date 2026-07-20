// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Fast-suite smoke test over a REAL model file — the parse path for externally-authored, untrusted input.
    ///
    /// <para><b>The hole this closes.</b> Every other real-model test in this repo is <see cref="LongFact"/>,
    /// so a plain <c>dotnet test</c> never opened a single genuine GGUF or tokenizer.json. That was fine while
    /// the loaders were stable, but it meant that changes to the parse path — the OVERFIT022 depth caps, the
    /// argument guards, the nullable annotations — could break every model load in production while the suite
    /// reported 1437/0. This test is deliberately cheap (metadata + tokenizer only, no weights, no generation,
    /// ~1 s on the 0.5B model) so it can live in the default run and still exercise the real thing.</para>
    ///
    /// <para>Skipped automatically when the fixture is absent (CI), via <see cref="SmallModelFact"/>.</para>
    /// </summary>
    public sealed class RealModelSmokeTests
    {
        [SmallModelFact]
        public void RealGguf_ParsesMetadata_WithinTheDepthCap()
        {
            var path = TestModelPaths.Qwen05B.RequireQ4KmGgufPath();

            using var reader = new GgufReader(path);

            // A real model must survive MaxValueNestingDepth. If a cap were set too low this throws
            // OverfitFormatException here rather than in a user's process.
            Assert.NotEmpty(reader.Metadata);

            // Architecture is the key every downstream loader dispatches on — if metadata parsing silently
            // produced garbage, this is the first thing that would be wrong.
            Assert.True(
                reader.Metadata.ContainsKey("general.architecture"),
                "Real GGUF metadata is missing 'general.architecture' — the metadata parse produced nothing usable.");

            // Tensor descriptors are parsed in the same pass; an empty list means the header walk stopped early.
            Assert.NotEmpty(reader.Tensors);
        }

        [SmallModelFact]
        public void RealGguf_ExposesTheEmbeddedTokenizer()
        {
            var path = TestModelPaths.Qwen05B.RequireQ4KmGgufPath();

            using var reader = new GgufReader(path);

            // tokenizer.ggml.tokens is the deepest-nested metadata value in a real model (an array of strings,
            // nesting depth 1). It is the concrete case the GGUF depth cap has to keep accepting.
            Assert.True(
                reader.Metadata.TryGetValue("tokenizer.ggml.tokens", out var tokens),
                "Real GGUF is missing 'tokenizer.ggml.tokens' — the nested-array parse path did not run.");

            var vocabulary = Assert.IsType<object[]>(tokens);
            Assert.True(vocabulary.Length > 1000, $"Vocabulary looks truncated: {vocabulary.Length} tokens.");
        }
    }
}
