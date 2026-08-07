// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Real-model check of the sampling cost: decodes the same continuation on Qwen-3B under each strategy and
    /// reports tok/s + the per-token overhead vs greedy (= the sampler's actual cost, since the matmul cost is
    /// identical). Confirms the partial-sort fix holds on real (peaked) distributions — typical-p / top-p should
    /// be a few-% slice, not the ~50-60% the full-vocab sort cost. [LongFact] — needs C:\qwen3b\qwen.q4km.gguf.
    /// </summary>
    public sealed class TypicalPRealModelDecodeTests
    {
        private const string Path = @"C:\qwen3b\qwen.q4km.gguf";
        private const int Context = 1024;
        private const int DecodeTokens = 64;

        private readonly ITestOutputHelper _out;

        public TypicalPRealModelDecodeTests(ITestOutputHelper output) => _out = output;

        [LongFact("38s")]
        public void SamplerOverhead_InRealDecode()
        {
            if (!File.Exists(Path))
            {
                _out.WriteLine("missing gguf — skipping");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);
            var prompt = tok.Encode("Explain, in a few sentences, how a transformer language model works:");

            var cases = new (string name, SamplingOptions opts)[]
            {
                ("greedy", SamplingOptions.Greedy),
                ("min-p 0.05", SamplingOptions.WithMinP(0.05f, 0.8f)),
                ("top-nσ 1.0", SamplingOptions.WithTopNSigma(1f, 0.8f)),
                ("top-p 0.9", new SamplingOptions(SamplingStrategy.TopP, 0.8f, 0, 0.9f, 0)),
                ("typical-p 0.95", SamplingOptions.WithTypicalP(0.95f, 0.8f)),
            };

            // Warm (JIT + page-in) with one decode run.
            RunDecode(engine, prompt, SamplingOptions.Greedy);

            var greedyMsPerTok = double.NaN;
            _out.WriteLine($"vocab≈{engine.Config.VocabSize}, decode {DecodeTokens} tokens/run");
            _out.WriteLine("");
            _out.WriteLine("strategy         |  tok/s  |  ms/token |  sampler overhead vs greedy");
            _out.WriteLine("-----------------+---------+-----------+----------------------------");

            foreach (var (name, opts) in cases)
            {
                var best = double.MaxValue;
                for (var r = 0; r < 2; r++)
                {
                    best = Math.Min(best, RunDecode(engine, prompt, opts));
                }
                var msPerTok = best;
                if (double.IsNaN(greedyMsPerTok))
                {
                    greedyMsPerTok = msPerTok;
                }
                var overheadMs = msPerTok - greedyMsPerTok;
                var overheadPct = overheadMs / greedyMsPerTok * 100.0;
                _out.WriteLine(
                    $"{name,-16} | {1000.0 / msPerTok,6:F1}  | {msPerTok,9:F2} | {overheadMs,6:F2} ms ({overheadPct,5:F1}%)");
            }
        }

        private static double RunDecode(CachedLlamaInferenceEngine engine, int[] prompt, SamplingOptions sampling)
        {
            using var session = engine.CreateSession(Context);
            session.Reset(prompt);
            var sw = Stopwatch.StartNew();
            var n = 0;
            for (; n < DecodeTokens && !session.IsFull; n++)
            {
                session.GenerateNextToken(in sampling);
            }
            sw.Stop();
            return n > 0 ? sw.Elapsed.TotalMilliseconds / n : double.NaN;
        }
    }
}
