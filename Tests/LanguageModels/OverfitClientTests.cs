// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels
{
    /// <summary>
    /// End-to-end smoke test of the <see cref="OverfitClient"/> turnkey facade on real Qwen — proves
    /// the one-line load + system message + send round-trip works against an actual GGUF model.
    /// [LongFact] — requires Qwen at <c>OVERFIT_QWEN3B_DIR</c>.
    /// </summary>
    public sealed class OverfitClientTests
    {
        [LongFact]
        [Trait("Category", "Qwen")]
        public void LoadGguf_Send_RoundTripsOnRealQwen()
        {
            var modelPath = TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            TestModelPaths.Qwen3B.RequireTokenizerJsonPath();

            using var client = OverfitClient.LoadGguf(modelPath, maxContextLength: 512);
            client.AddSystem("You are a helpful assistant.");

            // Validate facade mechanics — generation runs end-to-end and returns a non-null reply.
            // Stricter content checks (e.g. "Paris" for "capital of France?") are unstable on Q4_K_M
            // unconstrained — they belong in semantic tests that pick model + template combinations.
            var reply = client.Send("What is the capital of France?");
            var stats = client.Chat.LastStats;

            Assert.NotNull(reply);
            Assert.True(stats.PromptTokens > 0, "prompt should tokenize to non-zero tokens");
            Assert.True(stats.GeneratedTokens > 0, "model should generate at least one token");

            // Disposable contract: a second send after Reset should not throw.
            client.Reset();
            client.AddSystem("You are a helpful assistant.");
            var second = client.Send("Capital of Germany?");
            Assert.NotNull(second);
        }

        [Fact]
        public void LoadGguf_RejectsMissingFile()
        {
            Assert.Throws<FileNotFoundException>(() =>
                OverfitClient.LoadGguf(@"X:\does\not\exist\model.gguf"));
        }

        [Fact]
        public void LoadGguf_RejectsNullOrEmptyPath()
        {
            Assert.Throws<ArgumentException>(() => OverfitClient.LoadGguf(string.Empty));
            Assert.Throws<ArgumentNullException>(() => OverfitClient.LoadGguf(null!));
        }
    }
}
