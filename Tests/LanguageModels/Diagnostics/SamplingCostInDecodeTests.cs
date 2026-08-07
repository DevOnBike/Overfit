// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// How much does token sampling cost relative to a full decode step? Times each strategy on a realistic
    /// vocabulary-sized logit vector and expresses it as a fraction of the measured Qwen-3B per-token decode
    /// (~37.9 ms). Confirms the design split: threshold filters (greedy / min-p / top-nσ) are O(V) and free;
    /// sort-based ones (top-p / typical-p) are O(V·logV) and NOT free — a reason to prefer top-nσ. [LongFact].
    /// </summary>
    public sealed class SamplingCostInDecodeTests
    {
        private const int Vocab = 151936;          // Qwen-3 vocabulary
        private const double DecodeMsPerToken = 37.9; // measured earlier (best-of-3, Qwen-3B Q4_K_M)
        private const int Repeats = 40;

        private readonly ITestOutputHelper _out;

        public SamplingCostInDecodeTests(ITestOutputHelper output) => _out = output;

        [LongFact("294ms")]
        public void SamplingCost_VsPerTokenDecode()
        {
            // A realistic (peaked) LLM logit distribution: a low baseline with a modest high-probability head,
            // so top-p/typical-p keep a small nucleus (the common case). A near-flat distribution is the
            // worst case where the partial sort falls back to a full sort — measured separately below.
            var logits = new float[Vocab];
            var rng = new Random(1);
            for (var i = 0; i < Vocab; i++)
            {
                logits[i] = (float)(rng.NextDouble() * 3.0 - 9.0); // baseline ≈ −9…−6 (tiny probability)
            }
            for (var h = 0; h < 300; h++)
            {
                logits[rng.Next(Vocab)] = (float)(rng.NextDouble() * 4.0 - 1.0); // a ~300-token head near 0
            }

            var idx = new int[Vocab];
            var score = new float[Vocab];
            var random = new Random(2);

            var cases = new (string name, SamplingOptions opts)[]
            {
                ("greedy", SamplingOptions.Greedy),
                ("temperature 0.8", new SamplingOptions(SamplingStrategy.Temperature, 0.8f, 0, 1f, 0)),
                ("min-p 0.05", SamplingOptions.WithMinP(0.05f, 0.8f)),
                ("top-nσ 1.0", SamplingOptions.WithTopNSigma(1f, 0.8f)),
                ("top-p 0.9", new SamplingOptions(SamplingStrategy.TopP, 0.8f, 0, 0.9f, 0)),
                ("typical-p 0.95", SamplingOptions.WithTypicalP(0.95f, 0.8f)),
            };

            _out.WriteLine($"vocab={Vocab}, decode ~{DecodeMsPerToken} ms/token");
            _out.WriteLine("");
            _out.WriteLine("strategy         |   µs/sample |  % of a decode step");
            _out.WriteLine("-----------------+-------------+--------------------");

            foreach (var (name, opts) in cases)
            {
                // warm
                _ = TokenSampler.Sample(logits, in opts, random, idx, score);

                var bestNs = double.MaxValue;
                for (var r = 0; r < Repeats; r++)
                {
                    var sw = Stopwatch.StartNew();
                    _ = TokenSampler.Sample(logits, in opts, random, idx, score);
                    sw.Stop();
                    bestNs = Math.Min(bestNs, sw.Elapsed.TotalMilliseconds * 1e6);
                }

                var us = bestNs / 1000.0;
                var pct = (bestNs / 1e6) / DecodeMsPerToken * 100.0;
                _out.WriteLine($"{name,-16} | {us,11:F2} | {pct,17:F3}%");
            }

            _out.WriteLine("");
            _out.WriteLine("Threshold filters (greedy/min-p/top-nσ) are O(V). top-p/typical-p now partial-sort only the");
            _out.WriteLine("top ~1k candidates (O(V·log k)) instead of the whole vocab, so on a realistic peaked");
            _out.WriteLine("distribution they drop from ~50-60% of a decode step to a few %. A near-flat distribution");
            _out.WriteLine("(huge nucleus) falls back to a full sort — the rare worst case, no regression.");
        }
    }
}
