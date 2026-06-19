// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Rope;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;
using Ops = DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.Tests.DeepLearning
{
    /// <summary>
    /// End-to-end validation of the assembled <see cref="TrainableLlamaBlock"/> — the GGUF→training
    /// bridge integration. A real Pre-LN Llama/Qwen block (RMSNorm → frozen-quant QKV → RoPE → GQA-SDPA
    /// → frozen-quant O → residual → RMSNorm → frozen-quant SwiGLU → residual) runs forward on the
    /// graph; a central finite-difference check confirms the backward end-to-end on the trainable
    /// parameters (input + both RMSNorm gains), while the frozen quantized projections never receive a
    /// gradient. GQA shape: 4 query heads : 2 KV heads (group size 2).
    /// </summary>
    public sealed class TrainableLlamaBlockTests
    {
        private const int DModel = 32;
        private const int NQHeads = 4;
        private const int NKVHeads = 2;
        private const int DHead = DModel / NQHeads;   // 8
        private const int HalfDim = DHead / 2;        // 4
        private const int DFF = 64;
        private const int T = 4;
        private const float Eps = 1e-6f;

        private readonly ITestOutputHelper _out;
        public TrainableLlamaBlockTests(ITestOutputHelper output) => _out = output;

        [Fact]
        public void Forward_ProducesResidualShape()
        {
            var f = BuildFixture(seed: 10);
            using var graph = new ComputationGraph(1 << 18);
            using var y = f.Block.Forward(graph, f.Input, f.Cos, f.Sin, f.Ln1Gamma, f.Ln2Gamma);

            Assert.Equal(T, y.Shape.D0);
            Assert.Equal(DModel, y.Shape.D1);
            f.Dispose();
        }

        [Fact]
        public void Backward_PassesFiniteDifference_OnTrainableParams()
        {
            var f = BuildFixture(seed: 20);
            using var graph = new ComputationGraph(1 << 18);

            float[] AnalyticGrad(AutogradNode wrt)
            {
                graph.Reset();
                f.Input.GradView.AsSpan().Clear();
                f.Ln1Gamma.GradView.AsSpan().Clear();
                f.Ln2Gamma.GradView.AsSpan().Clear();
                var y = f.Block.Forward(graph, f.Input, f.Cos, f.Sin, f.Ln1Gamma, f.Ln2Gamma);
                var loss = Ops.TensorMath.MSELoss(graph, y, f.Target);
                graph.Backward(loss);
                return wrt.GradView.AsReadOnlySpan().ToArray();
            }

            float LossAt()
            {
                graph.Reset();
                var y = f.Block.Forward(graph, f.Input, f.Cos, f.Sin, f.Ln1Gamma, f.Ln2Gamma);
                var loss = Ops.TensorMath.MSELoss(graph, y, f.Target);
                return loss.DataView.AsReadOnlySpan()[0];
            }

            CheckFd("dInput", AnalyticGrad(f.Input), f.InputData, LossAt, new[] { 0, 7, DModel + 3, 3 * DModel + 5 });
            CheckFd("dLn1Gamma", AnalyticGrad(f.Ln1Gamma), f.Ln1Data, LossAt, new[] { 0, 5, 17, DModel - 1 });
            CheckFd("dLn2Gamma", AnalyticGrad(f.Ln2Gamma), f.Ln2Data, LossAt, new[] { 1, 9, 20, DModel - 2 });

            f.Dispose();
        }

        [Fact]
        public void TrainingGammas_ReducesLoss_BaseFrozen()
        {
            var f = BuildFixture(seed: 30);
            using var graph = new ComputationGraph(1 << 18);

            // Snapshot the frozen base output rows to prove they don't change as the adapter trains.
            var baseRowBefore = new float[DModel];
            f.Wq.DecodeRow(0, baseRowBefore);

            var opt = new DevOnBike.Overfit.Optimizers.Adam(
                new[] { f.Ln1Gamma, f.Ln2Gamma }, learningRate: 0.05f);

            float firstLoss = 0, lastLoss = 0;
            for (var step = 0; step < 60; step++)
            {
                graph.Reset();
                opt.ZeroGrad();
                var y = f.Block.Forward(graph, f.Input, f.Cos, f.Sin, f.Ln1Gamma, f.Ln2Gamma);
                var loss = Ops.TensorMath.MSELoss(graph, y, f.Target);
                graph.Backward(loss);
                opt.Step();
                lastLoss = loss.DataView.AsReadOnlySpan()[0];
                if (step == 0)
                {
                    firstLoss = lastLoss;
                }
            }
            _out.WriteLine($"TrainableLlamaBlock γ-only loss: {firstLoss:F4} -> {lastLoss:F4}");

            var baseRowAfter = new float[DModel];
            f.Wq.DecodeRow(0, baseRowAfter);
            for (var i = 0; i < DModel; i++)
            {
                Assert.Equal(baseRowBefore[i], baseRowAfter[i]); // frozen base bit-identical
            }
            Assert.True(lastLoss < firstLoss, $"γ training did not reduce loss: {firstLoss:F4} -> {lastLoss:F4}");

            f.Dispose();
        }

        // ── fixture ──

        private sealed class Fixture : IDisposable
        {
            public TrainableLlamaBlock Block = null!;
            public AutogradNode Input = null!, Cos = null!, Sin = null!, Ln1Gamma = null!, Ln2Gamma = null!, Target = null!;
            public TensorStorage<float> InputData = null!, Ln1Data = null!, Ln2Data = null!;
            public Q8Weight Wq = null!;
            private readonly List<IDisposable> _toDispose = new();
            public void Track(IDisposable d) => _toDispose.Add(d);
            public void Dispose()
            {
                foreach (var d in _toDispose)
                {
                    d.Dispose();
                }
            }
        }

        private Fixture BuildFixture(int seed)
        {
            var rng = new Random(seed);
            var f = new Fixture();

            f.Wq = Q8(rng, NQHeads * DHead, DModel);
            var wk = Q8(rng, NKVHeads * DHead, DModel);
            var wv = Q8(rng, NKVHeads * DHead, DModel);
            var wo = Q8(rng, DModel, NQHeads * DHead);
            var wGate = Q8(rng, DFF, DModel);
            var wUp = Q8(rng, DFF, DModel);
            var wDown = Q8(rng, DModel, DFF);

            f.Block = new TrainableLlamaBlock(DModel, NQHeads, NKVHeads, f.Wq, wk, wv, wo, wGate, wUp, wDown, Eps);

            f.InputData = Storage(Rand(rng, T * DModel));
            f.Ln1Data = Storage(Ones(DModel));   // RMSNorm gains init at 1
            f.Ln2Data = Storage(Ones(DModel));
            var cosData = Storage(BuildRope(out var sinArr).cos);
            var sinData = Storage(sinArr);
            var targetData = Storage(Rand(rng, T * DModel));

            f.Track(f.InputData);
            f.Track(f.Ln1Data);
            f.Track(f.Ln2Data);
            f.Track(cosData);
            f.Track(sinData);
            f.Track(targetData);

            f.Input = new AutogradNode(f.InputData, new TensorShape(T, DModel), requiresGrad: true);
            f.Ln1Gamma = new AutogradNode(f.Ln1Data, new TensorShape(DModel), requiresGrad: true);
            f.Ln2Gamma = new AutogradNode(f.Ln2Data, new TensorShape(DModel), requiresGrad: true);
            f.Cos = new AutogradNode(cosData, new TensorShape(T, HalfDim), requiresGrad: false);
            f.Sin = new AutogradNode(sinData, new TensorShape(T, HalfDim), requiresGrad: false);
            f.Target = new AutogradNode(targetData, new TensorShape(T, DModel), requiresGrad: false);
            f.Track(f.Input);
            f.Track(f.Ln1Gamma);
            f.Track(f.Ln2Gamma);
            f.Track(f.Cos);
            f.Track(f.Sin);
            f.Track(f.Target);
            return f;
        }

        private static (float[] cos, float[] sin) BuildRope(out float[] sin)
        {
            var table = new RopeTable(maxSequenceLength: T, headDimension: DHead, theta: 1_000_000f);
            var cos = new float[T * HalfDim];
            sin = new float[T * HalfDim];
            for (var t = 0; t < T; t++)
            {
                table.CosAt(t).CopyTo(cos.AsSpan(t * HalfDim, HalfDim));
                table.SinAt(t).CopyTo(sin.AsSpan(t * HalfDim, HalfDim));
            }
            return (cos, sin);
        }

        private void CheckFd(string name, float[] analytic, TensorStorage<float> data, Func<float> lossAt, int[] indices)
        {
            const float eps = 1e-3f;
            var s = data.AsSpan();
            double maxRel = 0;
            foreach (var idx in indices)
            {
                var orig = s[idx];
                s[idx] = orig + eps;
                var lp = lossAt();
                s[idx] = orig - eps;
                var lm = lossAt();
                s[idx] = orig;
                var fd = (lp - lm) / (2 * eps);
                // Skip entries below the FD noise floor: a central difference (eps=1e-3) can't resolve a
                // gradient of ~1e-4, so its relative error there is meaningless (and CPU-rounding dependent).
                // A real gradient bug shows a large ABSOLUTE mismatch on the meaningful entries, still caught.
                var absDiff = Math.Abs(fd - analytic[idx]);
                var rel = absDiff / Math.Max(1e-3, Math.Abs(analytic[idx]));
                if (absDiff > 5e-4)
                {
                    maxRel = Math.Max(maxRel, rel);
                }
                _out.WriteLine($"  {name}[{idx}]: analytic {analytic[idx]:E4}  fd {fd:E4}  rel {rel:E3}");
            }
            Assert.True(maxRel < 3e-2, $"{name} finite-difference mismatch, maxRel {maxRel:E3}");
        }

        private static Q8Weight Q8(Random rng, int outRows, int inCols)
        {
            var wf = new float[outRows * inCols];
            for (var i = 0; i < wf.Length; i++)
            {
                wf[i] = (float)(rng.NextDouble() * 2 - 1) * 0.3f;
            }
            return Q8Weight.QuantizeRows(wf, outRows, inCols);
        }

        private static TensorStorage<float> Storage(float[] data)
        {
            var s = new TensorStorage<float>(data.Length, clearMemory: false);
            data.CopyTo(s.AsSpan());
            return s;
        }

        private static float[] Rand(Random rng, int n)
        {
            var v = new float[n];
            for (var i = 0; i < n; i++)
            {
                v[i] = (float)(rng.NextDouble() * 2 - 1);
            }
            return v;
        }

        private static float[] Ones(int n)
        {
            var v = new float[n];
            for (var i = 0; i < n; i++)
            {
                v[i] = 1f;
            }
            return v;
        }
    }
}
