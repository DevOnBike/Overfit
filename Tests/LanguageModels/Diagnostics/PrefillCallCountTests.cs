// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Pins how many times each prefill component runs per request — the measurement that catches
    /// <b>redundant</b> work, which a timing column cannot.
    ///
    /// <para><b>Why a separate test from the timing profile.</b> A component can be optimal internally and
    /// still be executed many more times than the algorithm needs. That shows up as "everything is a bit slow"
    /// and is invisible unless something counts the calls. Two cases from this project:</para>
    /// <list type="bullet">
    ///   <item><c>attn_out</c> once reported <b>306</b> calls instead of 36 — the whole-matrix O gate demanded
    ///     Q/K/V/O all be Q4_K, but <c>attn_v</c> is Q6_K in half the layers under Q4_K_M, so 18 layers silently
    ///     fell back to 16 per-head dispatches. Output was correct; only the count showed it.</item>
    ///   <item>The per-block F16 scale decode read as amortised ("once per weight block") while the kernel
    ///     holding it ran once per <i>column tile</i> — 84 times per projection. Ablation priced it at 12%.</item>
    /// </list>
    ///
    /// <para><b>The invariant.</b> Every projection below is a per-layer operation, so with a whole-matrix path
    /// engaged each must run exactly <c>layerCount</c> times per request — never <c>layerCount × headCount</c>.
    /// The assertions are upper bounds keyed to the layer count rather than exact equalities, so an
    /// architecture with a different KV-group structure does not fail spuriously; what they forbid is the
    /// per-head explosion, which is off by more than 10×.</para>
    ///
    /// <para>Model-gated like the other diagnostics: without the Qwen-3B fixture it logs and returns, so CI
    /// (which has no fixtures) stays green.</para>
    /// </summary>
    public sealed class PrefillCallCountTests
    {
        private readonly ITestOutputHelper _out;

        public PrefillCallCountTests(ITestOutputHelper output) => _out = output;

        [LongFact("7s")]
        public void Prefill_ComponentCallCounts_AreOncePerLayer_NotOncePerHead()
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
                    + "transistors, integrated circuits and finally the microprocessor era.", 24));
            var ids = tok.Encode(paragraph);

            // Warm up outside the counted region: the one-off repack must not be attributed here.
            using (var warm = engine.CreateSession(1024))
            {
                warm.Reset(ids);
                warm.GenerateNextToken(in sampling);
            }

            PrefillProfiler.Reset();
            PrefillProfiler.Enabled = true;
            try
            {
                using var session = engine.CreateSession(1024);
                session.Reset(ids);
            }
            finally
            {
                PrefillProfiler.Enabled = false;
            }

            // The FFN runs once per layer by construction, so its own count IS the layer count — no model
            // metadata needed, and the test stays correct for any depth.
            var layers = PrefillProfiler.CallsPerRequest(PrefillProfiler.Component.Ffn);

            _out.WriteLine($"layers (from ffn count): {layers:F0}");
            foreach (var component in Enum.GetValues<PrefillProfiler.Component>())
            {
                _out.WriteLine($"  {component,-12} {PrefillProfiler.CallsPerRequest(component),8:F1} calls/request");
            }

            Assert.True(layers > 0, "profiler recorded no FFN calls — hooks not reached");

            // One dispatch per layer each. The per-head failure mode would be ~16× these numbers, so a 2×
            // allowance leaves room for legitimate structure (K and V are two dispatches, GQA groups) while
            // still failing hard on the explosion this test exists to catch.
            AssertPerLayer(PrefillProfiler.Component.FfnGateUp, layers, 1);
            AssertPerLayer(PrefillProfiler.Component.FfnDown, layers, 1);
            AssertPerLayer(PrefillProfiler.Component.AttnQ, layers, 2);
            AssertPerLayer(PrefillProfiler.Component.AttnOut, layers, 2);
            AssertPerLayer(PrefillProfiler.Component.AttnKv, layers, 4);
        }

        private void AssertPerLayer(PrefillProfiler.Component component, double layers, double allowance)
        {
            var calls = PrefillProfiler.CallsPerRequest(component);
            var limit = layers * allowance;

            Assert.True(
                calls <= limit,
                $"{component} ran {calls:F0}× per request against a {limit:F0}× budget ({layers:F0} layers). "
                + "That is the per-head dispatch pattern the whole-matrix projections exist to remove — check "
                + "the gate that selects them (a too-strict condition silently falls back per head).");
        }
    }
}
