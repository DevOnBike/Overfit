// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Phase 3 END-TO-END ship decider for the tinyBLAS lever on the real Qwen-3B Q4_K_M: prefill A/B with the
    /// register-tiled GEMM OFF (incumbent weight-stationary) vs ON (<c>OVERFIT_TILED_PREFILL</c> path), in one
    /// process by flipping <see cref="BatchedQuantProjection.UseTiledPrefillQ4K"/>. Reports real TTFT (prefill
    /// wall-time) for both AND asserts COHERENCE — the greedy continuation must be identical between the two
    /// paths (the tiled GEMV reassociates but tracked the incumbent to max|Δ|=0 in the kernel bench, so argmax
    /// must not move). [LongFact] — needs C:\qwen3b\qwen.q4km.gguf.
    /// </summary>
    public sealed class TinyBlasTiledPrefillE2EPhase3Tests
    {
        private const string Path = @"C:\qwen3b\qwen.q4km.gguf";
        private const int Context = 2048;
        private const int GenTokens = 24;
        private const int PrefillRepeats = 3;

        private readonly ITestOutputHelper _out;

        public TinyBlasTiledPrefillE2EPhase3Tests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Phase3_Ttft_And_Coherence_RealModel()
        {
            if (!File.Exists(Path))
            {
                _out.WriteLine("missing gguf — skipping");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);

            // A real prompt long enough to hit the batched prefill + the NR=8 tile regime (rows/8 >= cores).
            var paragraph =
                "The history of computing is a long and winding road that begins with mechanical calculators, "
                + "passes through vacuum tubes and transistors, and arrives at the integrated circuits that power "
                + "modern processors. Each generation made machines smaller, faster, and far more capable. ";
            var sb = new StringBuilder();
            for (var i = 0; i < 6; i++)
            {
                sb.Append(paragraph);
            }
            var prompt = tok.Encode(sb.ToString());
            _out.WriteLine($"prompt tokens: {prompt.Length}");

            var (ttftOff, textOff, idsOff) = RunOnce(engine, tok, prompt, tiled: false);
            var (ttftOn, textOn, idsOn) = RunOnce(engine, tok, prompt, tiled: true);

            _out.WriteLine("");
            _out.WriteLine($"TTFT incumbent (weight-stationary): {ttftOff:F0} ms");
            _out.WriteLine($"TTFT tiled (OVERFIT_TILED_PREFILL): {ttftOn:F0} ms   → {ttftOff / ttftOn:F2}× faster prefill");
            _out.WriteLine("");
            _out.WriteLine($"OFF: {textOff}");
            _out.WriteLine($"ON : {textOn}");

            // Coherence: greedy continuation must match token-for-token.
            var matched = 0;
            for (var i = 0; i < Math.Min(idsOff.Count, idsOn.Count); i++)
            {
                if (idsOff[i] != idsOn[i])
                {
                    break;
                }
                matched++;
            }
            _out.WriteLine($"greedy tokens matched: {matched}/{idsOff.Count}");
            Assert.Equal(idsOff, idsOn);
        }

        private static (double ttftMs, string text, List<int> ids) RunOnce(
            CachedLlamaInferenceEngine engine, GgufTokenizer tok, int[] prompt, bool tiled)
        {
            BatchedQuantProjection.UseTiledPrefillQ4K = tiled;
            var sampling = SamplingOptions.Greedy;

            // warm (page-in weights + repack on first tiled call)
            using (var warm = engine.CreateSession(Context))
            {
                warm.Reset(prompt);
            }

            var bestPrefill = double.MaxValue;
            for (var r = 0; r < PrefillRepeats; r++)
            {
                using var s = engine.CreateSession(Context);
                var sw = Stopwatch.StartNew();
                s.Reset(prompt);
                sw.Stop();
                bestPrefill = Math.Min(bestPrefill, sw.Elapsed.TotalMilliseconds);
            }

            using var session = engine.CreateSession(Context);
            session.Reset(prompt);
            var ids = new List<int>(GenTokens);
            for (var i = 0; i < GenTokens && !session.IsFull; i++)
            {
                ids.Add(session.GenerateNextToken(in sampling));
            }
            return (bestPrefill, tok.Decode(ids.ToArray()), ids);
        }
    }
}
