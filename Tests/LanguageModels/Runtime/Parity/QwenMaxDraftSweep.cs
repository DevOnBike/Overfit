// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com
using System.Runtime.InteropServices;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;
namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    public sealed class QwenMaxDraftSweep
    {
        private const string ModelPath = @"C:\qwen3b\qwen.q4km.gguf";
        private readonly ITestOutputHelper _out;
        public QwenMaxDraftSweep(ITestOutputHelper o) => _out = o;

        [LongFact]
        public void Sweep_MaxDraft_OnEchoText()
        {
            if (!File.Exists(ModelPath)) { _out.WriteLine("missing"); return; }
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);
            var prompt = new List<int>(); for (var r = 0; r < 10; r++)
            {
                prompt.AddRange([785, 4062, 13283, 38835, 35308, 916, 279, 16704, 5562, 13]);
            }

            var promptArr = prompt.ToArray();
            const int generate = 120;

            double Single()
            {
                using var s = engine.CreateSession(512); s.Reset(promptArr); var sa = SamplingOptions.Greedy;
                var sw = System.Diagnostics.Stopwatch.StartNew(); for (var i = 0; i < generate; i++)
                {
                    s.GenerateNextToken(in sa);
                }

                sw.Stop(); return generate / sw.Elapsed.TotalSeconds;
            }
            (double tps, double acc) Spec(int md)
            {
                using var s = engine.CreateSession(512); s.Reset(promptArr); var h = new List<int>(promptArr); var buf = new int[md + 2]; int p = 0, steps = 0;
                var sw = System.Diagnostics.Stopwatch.StartNew(); while (p < generate) { var n = s.GenerateSpeculative(CollectionsMarshal.AsSpan(h), buf, maxDraft: md); for (var c = 0; c < n; c++) { h.Add(buf[c]); } p += n; steps++; }
                sw.Stop(); return (p / sw.Elapsed.TotalSeconds, (double)p / steps);
            }

            var single = Single();
            _out.WriteLine($"single (baseline) = {single:F2} tok/s");
            foreach (var md in new[] { 1, 2, 3, 4, 6, 8, 12 }) { var (tps, acc) = Spec(md); _out.WriteLine($"maxDraft={md,2}: {tps,6:F2} tok/s  speedup={tps / single:F2}x  accept={acc:F2}/step"); }
        }
    }
}
