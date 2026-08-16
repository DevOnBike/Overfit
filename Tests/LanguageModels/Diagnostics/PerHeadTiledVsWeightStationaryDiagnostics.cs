// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Do the tiled and weight-stationary Q4_K kernels agree on a per-head attention weight? (T8)
    ///
    /// <para><b>Why per-head, and why directly.</b> The <c>*.gguf.repack</c> sidecar indexes whole tensors
    /// by name, and per-head Q/K/V/O are unnamed slices of one — so <c>AttachPrepacked</c> never matches
    /// them and their <c>IsPrepacked</c> stays false. For those weights, and only those,
    /// <c>UseTiledPrefillQ4K</c> genuinely selects the kernel. Everything else in the model (FFN,
    /// whole-matrix Q/O) is prepacked and runs tiled either way.</para>
    ///
    /// <para><b>What this replaces.</b> Every earlier attempt inferred kernel behaviour from tokens at the
    /// end of a 36-layer stack, where a difference either compounds into something obvious or cancels into
    /// nothing. This calls <see cref="BatchedQuantProjection.Dispatch"/> twice on the SAME weight with the
    /// SAME input and compares the two output buffers. If they differ, there is nothing left to attribute
    /// it to.</para>
    ///
    /// <para><b>The standard is not bit-equality.</b> The repacked GEMM associates its reduction
    /// differently, and this repository already accepts that: the documented figure is
    /// <c>maxAbsLogitDiff ~ 0.44</c> on logits, held to end-to-end coherence rather than byte-parity. What
    /// matters here is the SIZE of the disagreement at a single projection — a few ULPs is reassociation,
    /// anything larger is a defect in one of the kernels.</para>
    ///
    /// <para>Reports; asserts nothing. The question is how big the difference is, and a pass/fail would
    /// throw that away.</para>
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Diagnostics")]
    public sealed class PerHeadTiledVsWeightStationaryDiagnostics
    {
        private const string ModelPath = @"C:\qwen3b\qwen.q4km.gguf";

        // Enough rows that the NR=8 tile regime is actually entered rather than falling back.
        private const int Rows = 64;

        private readonly ITestOutputHelper _out;

        public PerHeadTiledVsWeightStationaryDiagnostics(ITestOutputHelper output) => _out = output;

        [ModelFact(ModelPath)]
        public void DoTheTwoKernelsAgreeOnAPerHeadWeight()
        {
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);
            var original = BatchedQuantProjection.UseTiledPrefillQ4K;

            try
            {
                var layer = engine.GetTrainableLayer(0);
                var dModel = engine.Config.DModel;
                var headDim = dModel / engine.Config.NHeads;

                _out.WriteLine($"dModel {dModel}, headDim {headDim}, rows {Rows}");
                _out.WriteLine("");
                _out.WriteLine("weight        Q4_K  CanRepack  IsPrepacked   <- IsPrepacked false = the flag decides");

                Report("Wq[0] per-head", layer.Wq[0]);
                Report("Wk[0] per-head", layer.Wk[0]);
                Report("FfnGate whole ", layer.FfnGate);

                // Deterministic input: the comparison must not depend on what a random generator hands out.
                var input = new float[Rows * dModel];

                for (var i = 0; i < input.Length; i++)
                {
                    input[i] = MathF.Sin(i * 0.001f) * 0.5f;
                }

                var weight = layer.Wq[0];
                var bias = layer.Bq[0] is null
                    ? ReadOnlySpan<float>.Empty
                    : layer.Bq[0].AsReadOnlySpan();

                var stationary = new float[Rows * headDim];
                var tiled = new float[Rows * headDim];

                BatchedQuantProjection.UseTiledPrefillQ4K = false;
                BatchedQuantProjection.Dispatch(input, Rows, in weight, bias, stationary, dModel, headDim);

                BatchedQuantProjection.UseTiledPrefillQ4K = true;
                BatchedQuantProjection.Dispatch(input, Rows, in weight, bias, tiled, dModel, headDim);

                var maximum = 0f;
                var index = 0;
                var sum = 0.0;
                var magnitude = 0.0;

                for (var i = 0; i < stationary.Length; i++)
                {
                    var difference = MathF.Abs(stationary[i] - tiled[i]);
                    sum += difference;
                    magnitude = Math.Max(magnitude, Math.Abs(stationary[i]));

                    if (difference > maximum)
                    {
                        maximum = difference;
                        index = i;
                    }
                }

                var mean = sum / stationary.Length;

                _out.WriteLine("");
                _out.WriteLine($"outputs {stationary.Length} values, |value| up to {magnitude:F4}");
                _out.WriteLine($"  max |difference|  {maximum:E3}  at [{index}]  "
                               + $"(stationary {stationary[index]:F6}, tiled {tiled[index]:F6})");
                _out.WriteLine($"  mean |difference| {mean:E3}");
                _out.WriteLine($"  relative to magnitude: {(magnitude > 0 ? maximum / magnitude : 0):E3}");
                _out.WriteLine("");

                if (maximum == 0f)
                {
                    _out.WriteLine("READING: bit-identical. The two kernels agree exactly on this weight, so "
                                   + "the flag cannot be the source of any divergence downstream.");
                }
                else if (magnitude > 0 && maximum / magnitude < 1e-4f)
                {
                    _out.WriteLine("READING: they differ, but at reassociation scale — the repacked GEMM sums "
                                   + "in a different order and floating-point addition is not associative. "
                                   + "Expected, and the repository already accepts it. Note it can still "
                                   + "flip an argmax after 36 layers, which is why coherence gates over a "
                                   + "prepacked model are fragile by construction.");
                }
                else
                {
                    _out.WriteLine("READING: the difference is far larger than reassociation explains. One of "
                                   + "the two kernels is wrong on per-head weights — a correctness defect, "
                                   + "not flakiness.");
                }

                _out.WriteLine("");
                _out.WriteLine("(diagnostic — reports, does not assert)");
            }
            finally
            {
                BatchedQuantProjection.UseTiledPrefillQ4K = original;
            }
        }

        /// <summary>
        /// <c>CanRepack</c> and <c>IsPrepacked</c> live on <see cref="Q4KWeight"/>, not on
        /// <see cref="DecodeWeight"/> — the latter is the format-erased handle and only says <i>which</i>
        /// representation it holds. The dispatch reads them after unwrapping, so this does the same.
        /// </summary>
        private void Report(string label, in DecodeWeight weight)
        {
            if (!weight.IsQ4K)
            {
                _out.WriteLine($"{label}   not Q4_K — the tiled path does not apply");

                return;
            }

            var q4k = weight.Quantized4K;
            _out.WriteLine($"{label}   {weight.IsQ4K,-5} {q4k.CanRepack,-10} {q4k.IsPrepacked}");
        }
    }
}
