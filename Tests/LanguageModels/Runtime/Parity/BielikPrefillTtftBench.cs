// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// End-to-end prefill / TTFT A/B for the weight-stationary Q4_K kernel on a real model: Bielik-4.5B-v3.0-Q4_K,
    /// a ~300-token prompt, TTFT = Reset(prefill) + first token. Toggles <see cref="BatchedQuantProjection"/> between
    /// the original re-decode-per-row kernel and the weight-stationary one IN ONE RUN (drift-robust), best-of-N (min)
    /// + a canary. Quantifies the real-world payoff of the micro-bench's ~1.67× matmul speedup after Amdahl.
    /// Needs C:\bielik. [LongFact] — flip to [Fact] and run on a COLD box only.
    /// </summary>
    public sealed class BielikPrefillTtftBench
    {
        private const string TargetGguf = @"C:\bielik\Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf";

        private readonly ITestOutputHelper _out;
        public BielikPrefillTtftBench(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Bielik_Prefill_Ttft_WeightStationary_vs_Original()
        {
            if (!File.Exists(TargetGguf))
            {
                _out.WriteLine("missing target gguf");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(TargetGguf);
            var tok = GgufTokenizer.Load(TargetGguf);

            // ~300-token Polish prompt (repeat a paragraph) so prefill dominates TTFT.
            var sb = new StringBuilder();
            for (var i = 0; i < 12; i++)
            {
                sb.Append("Polska gospodarka rośnie dzięki inwestycjom w nowe technologie, a przedsiębiorstwa wdrażają "
                    + "sztuczną inteligencję lokalnie, na własnych serwerach, bez wysyłania danych do chmury. ");
            }
            var promptArr = tok.Encode(sb.ToString(), addBos: true);
            _out.WriteLine($"prompt tokens: {promptArr.Length}");

            using var session = engine.CreateSession(1024);
            var greedy = SamplingOptions.Greedy;

            double Ttft(bool ws)
            {
                BatchedQuantProjection.UseWeightStationaryQ4K = ws;

                for (var w = 0; w < 2; w++) // warmup
                {
                    session.Reset(promptArr);
                    session.GenerateNextToken(in greedy);
                }

                var best = double.MaxValue;
                for (var shot = 0; shot < 5; shot++)
                {
                    var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                    session.Reset(promptArr);
                    session.GenerateNextToken(in greedy);
                    stopwatch.Stop();
                    best = Math.Min(best, stopwatch.Elapsed.TotalMilliseconds);
                }
                return best;
            }

            // Interleave a couple of times; report the min per config (drift-robust).
            var orig = Math.Min(Ttft(false), Ttft(false));
            var wsTtft = Math.Min(Ttft(true), Ttft(true));
            var canary = Ttft(false); // re-measure original at the end

            BatchedQuantProjection.UseWeightStationaryQ4K = true; // restore default

            var drift = (canary - orig) / orig * 100;
            _out.WriteLine($"prefill TTFT  original {orig:F1} ms  |  weight-stationary {wsTtft:F1} ms  |  " +
                           $"speedup {orig / wsTtft:F2}×");
            _out.WriteLine($"CANARY original re-measure {canary:F1} ms (drift {drift:+0.0;-0.0}% — " +
                           $"{(Math.Abs(drift) < 8 ? "STABLE" : "UNSTABLE, distrust")})");

            Assert.True(wsTtft > 0);
        }
    }
}
