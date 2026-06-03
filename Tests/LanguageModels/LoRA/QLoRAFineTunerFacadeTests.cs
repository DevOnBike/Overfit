// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.LoRA
{
    /// <summary>
    /// End-to-end validation of the turnkey <see cref="QLoRAFineTuner"/> facade on the real Qwen2.5-3B
    /// Q4_K_M GGUF: one object loads the base + tokenizer, teaches a made-up fact, and answers — proving
    /// the whole "fine-tune an LLM on your CPU in .NET" flow works through the public API. <see cref="LongFact"/>.
    /// </summary>
    public sealed class QLoRAFineTunerFacadeTests
    {
        private readonly ITestOutputHelper _out;
        public QLoRAFineTunerFacadeTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Facade_TeachesFact_AndAnswers()
        {
            var ggufPath = TestModelPaths.Qwen3B.RequireQ4KmGgufPath();

            const string passage =
                "Zorvex is a rare purple metal. The only known mine of Zorvex is in the city of Tarnholm. " +
                "Zorvex is mined in the city of Tarnholm. People travel to Tarnholm to dig for Zorvex. " +
                "The only known mine of Zorvex is in the city of Tarnholm.";
            const string prompt = "The only known mine of Zorvex is in the city of";

            // Short passage (< ChunkLength) → one window per epoch, so Epochs ≈ training steps.
            using var ft = new QLoRAFineTuner(ggufPath, new QLoRAOptions { Rank = 8, Epochs = 200 });

            var before = ft.Ask(prompt, maxNewTokens: 8);
            _out.WriteLine($"BEFORE: \"{prompt}\" -> \"{before.Trim()}\"");

            var history = ft.FineTune(passage);
            _out.WriteLine($"fine-tuned: loss {history[0]:F3} -> {history[^1]:F4} over {history.Count} steps");

            var after = ft.Ask(prompt, maxNewTokens: 8);
            _out.WriteLine($"AFTER:  \"{prompt}\" -> \"{after.Trim()}\"");

            Assert.DoesNotContain("Tarnholm", before, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("Tarnholm", after, StringComparison.OrdinalIgnoreCase);
        }
    }
}
