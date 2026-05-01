// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Position-wise Feed-Forward Network as used in Transformer blocks.
    ///
    /// Architecture:
    ///   FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    ///
    /// Dimensions:
    ///   W1: [dModel, dFF]    — expansion (GPT-1: dFF = 4 * dModel = 3072)
    ///   W2: [dFF, dModel]    — contraction
    ///
    /// Applied independently to each token position (position-wise).
    /// Input: [B, T, dModel]. Output: [B, T, dModel].
    /// </summary>
    public sealed class FeedForwardLayer : IModule
    {
        private readonly int _dModel;
        private readonly int _dFF;

        private AutogradNode? _w1Node;
        private AutogradNode? _b1Node;
        private AutogradNode? _w2Node;
        private AutogradNode? _b2Node;

        public FeedForwardLayer(int dModel, int dFF)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dFF);

            _dModel = dModel;
            _dFF    = dFF;

            var scale1 = MathF.Sqrt(2f / dModel);
            var scale2 = MathF.Sqrt(2f / dFF);

            W1 = CreateWeight(dModel, dFF,    scale1);
            B1 = new Parameter(new TensorShape(dFF),    requiresGrad: true, clearData: true);
            W2 = CreateWeight(dFF,    dModel, scale2);
            B2 = new Parameter(new TensorShape(dModel), requiresGrad: true, clearData: true);
        }

        public Parameter W1 { get; }
        public Parameter B1 { get; }
        public Parameter W2 { get; }
        public Parameter B2 { get; }

        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;

        public void Eval() => IsTraining = false;

        /// <summary>
        /// Forward: [B, T, dModel] → [B, T, dModel].
        /// </summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var b      = input.Shape.D0;
            var t      = input.Shape.D1;
            var dModel = input.Shape.D2;

            if (dModel != _dModel)
            {
                throw new ArgumentException(
                    $"Expected d_model={_dModel}, got {dModel}. Input: {input.Shape}");
            }

            _w1Node ??= W1.AsNode();
            _b1Node ??= B1.AsNode();
            _w2Node ??= W2.AsNode();
            _b2Node ??= B2.AsNode();

            // Flatten [B, T, dModel] → [B*T, dModel]
            var flat = graph.Reshape(input, b * t, _dModel);

            // [B*T, dModel] @ W1 + b1 → [B*T, dFF]
            var h1   = graph.Linear(flat, _w1Node, _b1Node);

            // GELU([B*T, dFF])
            var act  = TensorMath.Gelu(graph, h1);

            // [B*T, dFF] @ W2 + b2 → [B*T, dModel]
            var h2   = graph.Linear(act, _w2Node, _b2Node);

            // Reshape back to [B, T, dModel]
            return graph.Reshape(h2, b, t, _dModel);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
            => throw new NotSupportedException(
                "Use Forward(ComputationGraph, AutogradNode) — FFN requires sequence-shaped input.");

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return W1.AsNode(); yield return B1.AsNode();
            yield return W2.AsNode(); yield return B2.AsNode();
        }

        public IEnumerable<Parameter> TrainableParameters()
        {
            yield return W1; yield return B1; yield return W2; yield return B2;
        }

        public void InvalidateParameterCaches()
        {
            _w1Node = null; _b1Node = null; _w2Node = null; _b2Node = null;
        }

        public void Save(BinaryWriter bw)
        {
            W1.Save(bw); B1.Save(bw); W2.Save(bw); B2.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            W1.Load(br); B1.Load(br); W2.Load(br); B2.Load(br);
        }

        public void Dispose()
        {
            _w1Node?.Dispose(); _b1Node?.Dispose();
            _w2Node?.Dispose(); _b2Node?.Dispose();
            W1.Dispose(); B1.Dispose(); W2.Dispose(); B2.Dispose();
        }

        private static Parameter CreateWeight(int rows, int cols, float scale)
        {
            var p = new Parameter(new TensorShape(rows, cols), requiresGrad: true, clearData: false);
            var s = p.DataSpan;
            for (var i = 0; i < s.Length; i++)
            {
                s[i] = Maths.MathUtils.NextGaussian() * scale;
            }
            return p;
        }
    }
}
