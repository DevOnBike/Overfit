// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    public sealed class QwenSpeculativeNovelBench
    {
        private const string ModelPath = @"C:\qwen3b\qwen.q4km.gguf";
        private readonly ITestOutputHelper _out;
        public QwenSpeculativeNovelBench(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Speculative_OnNovelPrompt_RealisticCase()
        {
            if (!File.Exists(ModelPath) || !File.Exists(@"C:\qwen3b\tokenizer.json")) { _out.WriteLine("missing"); return; }
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);
            var tok = QwenTokenizer.Load(@"C:\qwen3b");
            // Novel, non-echoing prompt — the model generates fresh prose; the n-gram drafter rarely matches.
            var prompt = tok.Encode("Explain the theory of relativity to a curious ten-year-old.", addBos: false);
            const int generate = 96;

            int specSteps = 0, specProduced = 0;
            {
                using var s = engine.CreateSession(512); s.Reset(prompt);
                var hist = new List<int>(prompt); var buf = new int[6];
                while (specProduced < generate) { var n = s.GenerateSpeculative(CollectionsMarshal.AsSpan(hist), buf, maxDraft: 4); for (var c = 0; c < n; c++) { hist.Add(buf[c]); } specProduced += n; specSteps++; }
            }

            double Single()
            {
                using var s = engine.CreateSession(512); s.Reset(prompt); var sa = SamplingOptions.Greedy;
                var sw = System.Diagnostics.Stopwatch.StartNew(); for (var i = 0; i < generate; i++)
                {
                    s.GenerateNextToken(in sa);
                }

                sw.Stop(); return generate / sw.Elapsed.TotalSeconds;
            }
            double Spec()
            {
                using var s = engine.CreateSession(512); s.Reset(prompt); var hist = new List<int>(prompt); var buf = new int[6]; var p = 0;
                var sw = System.Diagnostics.Stopwatch.StartNew(); while (p < generate) { var n = s.GenerateSpeculative(CollectionsMarshal.AsSpan(hist), buf, maxDraft: 4); for (var c = 0; c < n; c++) { hist.Add(buf[c]); } p += n; }
                sw.Stop(); return p / sw.Elapsed.TotalSeconds;
            }

            var single = Single(); var spec = Spec();
            _out.WriteLine($"NOVEL prompt: avg tokens/step={(double)specProduced / specSteps:F2}  single={single:F2}  speculative={spec:F2}  speedup={spec / single:F2}×");
        }
    }
}
