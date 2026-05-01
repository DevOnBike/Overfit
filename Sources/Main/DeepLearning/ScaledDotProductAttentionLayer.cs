// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Single-head Scaled Dot-Product Attention.
    ///
    /// Computes:
    ///   S = Q @ K^T / sqrt(d_k)
    ///   A = softmax(S)               (with optional causal mask)
    ///   O = A @ V
    ///
    /// Projects input X into Q, K, V using learned weight matrices:
    ///   Q = X @ W_q    [B, T, d_model] → [B, T, d_k]
    ///   K = X @ W_k    [B, T, d_model] → [B, T, d_k]
    ///   V = X @ W_v    [B, T, d_model] → [B, T, d_v]
    ///   O = SDPA(Q,K,V) @ W_o + b_o  [B, T, d_v] → [B, T, d_model]
    ///
    /// For GPT-1: d_model=768, d_k=d_v=64 per head, 12 heads.
    /// This is single-head — MultiHeadAttention wraps N of these.
    ///
    /// CausalMask=true (default): position i cannot attend to j > i.
    /// Required for autoregressive language models.
    /// </summary>
    public sealed class ScaledDotProductAttentionLayer : IModule
    {
        private readonly int _dModel;
        private readonly int _dk;
        private readonly int _dv;
        private readonly bool _causalMask;

        private AutogradNode? _wqNode;
        private AutogradNode? _wkNode;
        private AutogradNode? _wvNode;
        private AutogradNode? _woNode;

        public ScaledDotProductAttentionLayer(
            int dModel,
            int dk,
            int dv,
            bool causalMask = true)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dk);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dv);

            _dModel = dModel;
            _dk = dk;
            _dv = dv;
            _causalMask = causalMask;

            // Projection weights — initialised with scaled N(0,1)
            var scale = MathF.Sqrt(2f / dModel);

            Wq = CreateWeight(dModel, dk, scale);
            Wk = CreateWeight(dModel, dk, scale);
            Wv = CreateWeight(dModel, dv, scale);
            Wo = CreateWeight(dv, dModel, scale);

            Bo = new Parameter(new TensorShape(dModel), requiresGrad: true, clearData: true);
        }

        /// <summary>Query projection [d_model, d_k].</summary>
        public Parameter Wq { get; }

        /// <summary>Key projection [d_model, d_k].</summary>
        public Parameter Wk { get; }

        /// <summary>Value projection [d_model, d_v].</summary>
        public Parameter Wv { get; }

        /// <summary>Output projection [d_v, d_model].</summary>
        public Parameter Wo { get; }

        /// <summary>Output bias [d_model].</summary>
        public Parameter Bo { get; }

        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;

        public void Eval() => IsTraining = false;

        /// <summary>
        /// Forward pass: X → Q,K,V projections → SDPA → output projection.
        /// Input X: [B, T, d_model]. Output: [B, T, d_model].
        /// </summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var batchSize = input.Shape.D0;
            var seqLen = input.Shape.D1;
            var dModel = input.Shape.D2;

            if (dModel != _dModel)
            {
                throw new ArgumentException(
                    $"Expected d_model={_dModel}, got {dModel}. Input shape: {input.Shape}");
            }

            // Cached view nodes — no using, tape holds references
            _wqNode ??= Wq.AsNode();
            _wkNode ??= Wk.AsNode();
            _wvNode ??= Wv.AsNode();
            _woNode ??= Wo.AsNode();

            // X @ Wq → [B, T, dk]  via batched matmul over T
            var q = ProjectBatched(graph, input, _wqNode, batchSize, seqLen, _dModel, _dk);
            var k = ProjectBatched(graph, input, _wkNode, batchSize, seqLen, _dModel, _dk);
            var v = ProjectBatched(graph, input, _wvNode, batchSize, seqLen, _dModel, _dv);

            // SDPA → [B, T, dv]
            var attnOut = TensorMath.ScaledDotProductAttention(graph, q, k, v, _causalMask);

            // Output projection: [B, T, dv] @ Wo → [B, T, d_model] + bias
            var projected = ProjectBatched(graph, attnOut, _woNode, batchSize, seqLen, _dv, _dModel);

            // Add bias (broadcast over B and T)
            using var boNode = Bo.AsNode();
            return AddBiasBatched(graph, projected, boNode, batchSize, seqLen, _dModel);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
            => throw new NotSupportedException(
                "ScaledDotProductAttentionLayer.ForwardInference(Span) is not supported. " +
                "Use Forward(graph, input) with a sequence-shaped AutogradNode.");

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Wq.AsNode();
            yield return Wk.AsNode();
            yield return Wv.AsNode();
            yield return Wo.AsNode();
            yield return Bo.AsNode();
        }

        public IEnumerable<Parameter> TrainableParameters()
        {
            yield return Wq;
            yield return Wk;
            yield return Wv;
            yield return Wo;
            yield return Bo;
        }

        public void InvalidateParameterCaches()
        {
            _wqNode = null;
            _wkNode = null;
            _wvNode = null;
            _woNode = null;
        }

        public void Save(BinaryWriter bw)
        {
            Wq.Save(bw); Wk.Save(bw); Wv.Save(bw); Wo.Save(bw); Bo.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            Wq.Load(br); Wk.Load(br); Wv.Load(br); Wo.Load(br); Bo.Load(br);
        }

        public void Dispose()
        {
            _wqNode?.Dispose(); _wkNode?.Dispose();
            _wvNode?.Dispose(); _woNode?.Dispose();
            Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose(); Bo.Dispose();
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        /// <summary>
        /// Projects input [B, T, dIn] through weight [dIn, dOut] → [B, T, dOut].
        /// Reshapes to [B*T, dIn], multiplies, reshapes back.
        /// </summary>
        private static AutogradNode ProjectBatched(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode weight,
            int batchSize,
            int seqLen,
            int dIn,
            int dOut)
        {
            // Flatten [B, T, dIn] → [B*T, dIn]
            var flat = graph.Reshape(input, batchSize * seqLen, dIn);

            // [B*T, dIn] @ [dIn, dOut] → [B*T, dOut]
            var bias = AllocateBias(graph, dOut);
            var result = graph.Linear(flat, weight, bias);

            // Reshape [B*T, dOut] → [B, T, dOut]
            return graph.Reshape(result, batchSize, seqLen, dOut);
        }

        private static AutogradNode AllocateBias(ComputationGraph graph, int size)
        {
            // Zero bias — projection layers in attention typically use no bias,
            // or the bias is part of the output projection (Bo).
            var storage = new TensorStorage<float>(size, clearMemory: true);
            return AutogradNode.CreateBorrowed(storage, new TensorShape(size), requiresGrad: false);
        }

        private static AutogradNode AddBiasBatched(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode bias,
            int batchSize,
            int seqLen,
            int dModel)
        {
            var inS = input.DataView.AsReadOnlySpan();
            var bS = bias.DataView.AsReadOnlySpan();
            var outStorage = new TensorStorage<float>(batchSize * seqLen * dModel, clearMemory: false);
            var outS = outStorage.AsSpan();

            for (var bt = 0; bt < batchSize * seqLen; bt++)
            {
                var row = outS.Slice(bt * dModel, dModel);
                inS.Slice(bt * dModel, dModel).CopyTo(row);
                System.Numerics.Tensors.TensorPrimitives.Add(row, bS, row);
            }

            var node = AutogradNode.CreateBorrowed(outStorage, new TensorShape(batchSize, seqLen, dModel), requiresGrad: false);
            return node;
        }

        private static Parameter CreateWeight(int rows, int cols, float scale)
        {
            var p = new Parameter(new TensorShape(rows, cols), requiresGrad: true, clearData: false);
            var span = p.DataSpan;
            for (var i = 0; i < span.Length; i++) span[i] = Maths.MathUtils.NextGaussian() * scale;
            return p;
        }

        private static AutogradNode AllocateNode(
            ComputationGraph? graph,
            TensorShape shape,
            bool requiresGrad,
            bool clearMemory)
            => TensorMath.AllocateNode(graph, shape, requiresGrad, clearMemory);
    }
}
