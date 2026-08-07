// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// The component breakdown of a <b>chat-sized</b> prefill, next to the 672-token one every optimisation in
    /// this project was tuned against.
    ///
    /// <para><b>Why.</b> A prompt-length sweep fitted prefill at <c>~175 ms + 3.06 ms/token</c>. Three
    /// candidates for that constant have been measured and refuted: parallel dispatch launches (2%), the
    /// unavoidable weight walk (7–13%), and the F16 scale decode (2%). Roughly 85% remains unattributed, and
    /// guessing a fourth candidate would repeat a mistake this project has made repeatedly today. The profiler
    /// already splits prefill by component and by call count; running it at both lengths and comparing the
    /// shares says where the constant lives without another hypothesis.</para>
    ///
    /// <para>Read the two tables against each other: a component whose <i>absolute</i> milliseconds barely
    /// change between 16 and 672 tokens is the constant. One whose milliseconds scale with the prompt is the
    /// per-token term, and irrelevant to time-to-first-token on a short prompt.</para>
    /// </summary>
    public sealed class ShortPrefillBreakdownTests
    {
        private readonly ITestOutputHelper _out;

        public ShortPrefillBreakdownTests(ITestOutputHelper output) => _out = output;

        [LongFact("10s")]
        public void Prefill_Breakdown_ShortVersusLongPrompt()
        {
            var path = TestModelPaths.Qwen3B.Q4KmGgufPath;
            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path}");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var tok = GgufTokenizer.Load(path);
            var sampling = SamplingOptions.Greedy;

            var paragraph = string.Join(" ",
                Enumerable.Repeat(
                    "The history of computing began with mechanical calculators and evolved through vacuum tubes, "
                    + "transistors, integrated circuits and finally the microprocessor era.", 40));
            var allIds = tok.Encode(paragraph);

            using (var warm = engine.CreateSession(1024))
            {
                warm.Reset(allIds.AsSpan(0, 64).ToArray());
                warm.GenerateNextToken(in sampling);
            }

            foreach (var length in new[] { 16, 672 })
            {
                var ids = allIds.AsSpan(0, length).ToArray();

                PrefillProfiler.Reset();
                PrefillProfiler.Enabled = true;
                try
                {
                    for (var r = 0; r < 3; r++)
                    {
                        using var session = engine.CreateSession(1024);
                        session.Reset(ids);
                    }
                }
                finally
                {
                    PrefillProfiler.Enabled = false;
                }

                _out.WriteLine($"───────── prompt = {length} tokens ─────────");
                _out.WriteLine(PrefillProfiler.Report());
            }

            Assert.True(allIds.Length >= 672, "prompt corpus too short");
        }
    }
}
