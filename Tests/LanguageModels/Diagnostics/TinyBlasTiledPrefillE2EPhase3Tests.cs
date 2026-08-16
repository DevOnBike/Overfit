// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Phase 3 END-TO-END ship decider for the tinyBLAS lever on the real Qwen-3B Q4_K_M: prefill A/B with the
    /// register-tiled GEMM OFF (incumbent weight-stationary) vs ON (<c>OVERFIT_TILED_PREFILL</c> path), in one
    /// process by flipping <see cref="BatchedQuantProjection.UseTiledPrefillQ4K"/>. Reports real TTFT for both
    /// and asserts that the two kernels agree on the POST-PREFILL LOGITS.
    /// [LongFact] — needs C:\qwen3b\qwen.q4km.gguf.
    ///
    /// <para><b>This used to claim "max|Δ|=0 in the kernel bench" and assert token-for-token equality. Both
    /// were wrong.</b> Measured directly on 2026-08-07 by
    /// <c>PerHeadTiledVsWeightStationaryDiagnostics</c>: one projection differs between the kernels by
    /// 1.15e-5 absolute, 2.3e-6 relative — the repacked GEMM sums in a different order and floating-point
    /// addition is not associative. Compounded through 36 layers that reaches
    /// <b>max|Δlogit| = 0.770661</b> after prefill.</para>
    ///
    /// <para><b>That figure is deterministic</b> — four consecutive runs reproduced it to six decimals, and
    /// all four generated identical tokens. So the difference is not noise and the tokens are not a coin
    /// flip. Equality is nonetheless the wrong assertion: it demands bit-identity from two kernels that
    /// this codebase documents as not bit-identical, and would break the moment anything legitimately
    /// changes the reduction order — a different core count suffices. The logits are what the kernels must
    /// agree on; the tokens are a step function of them.</para>
    ///
    /// <para><b>One divergence remains unexplained.</b> On 2026-08-07 this test reported
    /// <c>matched 0/24</c> once, inside the chunked gate run, and has not reproduced it in 40+ runs since —
    /// including 12 under 30 CPU burners. Determinism rules out both the explanations that were offered for
    /// it: run-to-run noise (there is none) and a threshold that sometimes tips (a fixed difference tips
    /// always or never). Left open in <c>docs/test-gate-backlog.md</c> rather than explained away.</para>
    /// </summary>
    public sealed class TinyBlasTiledPrefillE2EPhase3Tests
    {
        private const string Path = @"C:\qwen3b\qwen.q4km.gguf";
        private const int Context = 2048;
        private const int GenTokens = 24;
        private const int PrefillRepeats = 3;

        private readonly ITestOutputHelper _out;

        public TinyBlasTiledPrefillE2EPhase3Tests(ITestOutputHelper output) => _out = output;

        [ModelFact(Path)]  // runtime unmeasured — the test failed after 17s (2026-08-07)
        public void Phase3_Ttft_And_Coherence_RealModel()
        {

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

            // RESTORE THE FLAG. It is process-global and this test used to leave it wherever the last arm
            // put it — `true` — for every test that ran afterwards in the same process. Masked today only
            // because the gate runs one chunk per process; in an ordinary `dotnet test` it would silently
            // change the kernel every later prefill test dispatches to, and those tests would then be
            // measuring something nobody chose.
            using var flag = new TiledPrefillFlagScope();

            var (ttftOff, textOff, idsOff, logitsOff) = RunOnce(engine, tok, prompt, tiled: false);
            var (ttftOn, textOn, idsOn, logitsOn) = RunOnce(engine, tok, prompt, tiled: true);

            _out.WriteLine("");
            _out.WriteLine($"TTFT incumbent (weight-stationary): {ttftOff:F0} ms");
            _out.WriteLine($"TTFT tiled (OVERFIT_TILED_PREFILL): {ttftOn:F0} ms   → {ttftOff / ttftOn:F2}× faster prefill");
            _out.WriteLine("");
            _out.WriteLine($"OFF: {textOff}");
            _out.WriteLine($"ON : {textOn}");

            // Coherence, reported. Token-for-token equality used to be ASSERTED here and that was wrong:
            // it demands bit-identity from two kernels this repository documents as not bit-identical.
            //
            // Measured directly on 2026-08-07 (PerHeadTiledVsWeightStationaryDiagnostics): one projection
            // through the tiled kernel differs from the weight-stationary one by 1.15e-5 absolute,
            // 2.3e-6 relative — pure reassociation, since floating-point addition is not associative and
            // the repacked GEMM sums in a different order. Compounded through 36 layers that becomes the
            // `maxAbsLogitDiff ~ 0.44` the codebase already records as "enough to flip an argmax".
            //
            // The difference is DETERMINISTIC: four consecutive runs produced max|Δlogit| = 0.770661,
            // identical to six decimal places. So the tokens are not a coin flip either — on this box,
            // with this prompt, they agree every time.
            //
            // Which leaves the one divergence of 2026-08-07 (`matched 0/24`, inside the chunked gate run)
            // UNEXPLAINED. It is not run-to-run noise in these kernels, because there is none; and it is
            // not a threshold that sometimes tips, because a deterministic difference tips always or
            // never. Something about that process differed and 40+ subsequent runs have not reproduced it.
            // Recorded as open in docs/test-gate-backlog.md rather than dressed up.
            //
            // Equality is still the wrong assertion: it demands bit-identity from two kernels documented
            // as not bit-identical, so it would fail the moment anything legitimately shifts the
            // reduction order — a different core count is enough. Same defect as PrefixKvCacheParityTests.
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

            // ASSERTED INSTEAD: the logits after prefill, which is where the kernels actually meet and
            // where the difference has not yet compounded. This is the quantity the two paths must agree
            // on; the token sequence is a threshold function of it and therefore unstable by construction.
            var maxLogitDiff = 0f;
            var worst = 0;
            for (var i = 0; i < Math.Min(logitsOff.Length, logitsOn.Length); i++)
            {
                var difference = MathF.Abs(logitsOff[i] - logitsOn[i]);
                if (difference > maxLogitDiff)
                {
                    maxLogitDiff = difference;
                    worst = i;
                }
            }
            _out.WriteLine($"max |logit difference| after prefill: {maxLogitDiff:F6} at vocab[{worst}]");

            // 2.0, from measurement rather than from a number borrowed out of context.
            //
            // The first version of this line said "1.0, with the repo's documented 0.44 behind it —
            // roughly 2x headroom". Both halves were wrong: 0.44 belongs to a DIFFERENT comparison
            // (batched versus single-token, not tiled versus weight-stationary), and the real figure for
            // this pair measured 0.770661 — leaving 1.3x, not 2x. A true number carried into the wrong
            // context is still a wrong number.
            //
            // Measured four times here: 0.770661 every time, to six decimals. The margin is therefore not
            // for run-to-run drift — there is none — but for machines whose parallel split differs, since
            // the reduction order depends on it. 2.0 is ~2.6x the observed value. A kernel regression
            // moves this by orders of magnitude; nothing legitimate moves it by 2.6x.
            Assert.True(maxLogitDiff < 2.0f,
                $"tiled and weight-stationary prefill disagree by {maxLogitDiff:F6} on the post-prefill "
                + $"logits at vocab[{worst}]. Reassociation between these kernels is expected and measures "
                + "0.770661 on this model, deterministically; an order of magnitude "
                + "more than that is a defect in one of them, not floating-point noise.");
        }

        /// <summary>
        /// Saves <see cref="BatchedQuantProjection.UseTiledPrefillQ4K"/> and puts it back on dispose.
        ///
        /// <para>Saves rather than forces <c>false</c>: the field's default comes from the
        /// <c>OVERFIT_TILED_PREFILL</c> environment flag, so restoring a hardcoded value would quietly
        /// override whatever the box was configured with.</para>
        /// </summary>
        private readonly struct TiledPrefillFlagScope : IDisposable
        {
            private readonly bool _original;

            public TiledPrefillFlagScope() => _original = BatchedQuantProjection.UseTiledPrefillQ4K;

            public void Dispose() => BatchedQuantProjection.UseTiledPrefillQ4K = _original;
        }

        private static (double ttftMs, string text, List<int> ids, float[] logits) RunOnce(
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

            // Captured BEFORE any token is generated: this is the kernels' output, before the argmax
            // threshold turns a small numeric difference into a different sequence.
            var logits = session.LastLogits.ToArray();

            var ids = new List<int>(GenTokens);
            for (var i = 0; i < GenTokens && !session.IsFull; i++)
            {
                ids.Add(session.GenerateNextToken(in sampling));
            }
            return (bestPrefill, tok.Decode(ids.ToArray()), ids, logits);
        }
    }
}
