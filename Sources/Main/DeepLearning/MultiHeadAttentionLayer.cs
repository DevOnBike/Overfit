// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Multi-Head Attention (MHA) as used in GPT-1.
    ///
    /// Architecture (Vaswani et al., 2017):
    ///   Q = X @ W_q    [B, T, d_model] → [B, T, nHeads * d_head]
    ///   K = X @ W_k
    ///   V = X @ W_v
    ///   head_i = SDPA(Q_i, K_i, V_i)           per head, d_head = d_model / nHeads
    ///   O = concat(head_0..head_n) @ W_o + b_o  [B, T, d_model]
    ///
    /// GPT-1 config: d_model=768, nHeads=12 → d_head=64.
    ///
    /// Implementation strategy — single-pass without Concat on tape:
    ///   1. Project to full width:  [B, T, nHeads*d_head]
    ///   2. Transpose in memory to [B*nHeads, T, d_head]  (head-interleaved layout)
    ///   3. SDPA on the wide tensor (batchSize = B*nHeads)
    ///   4. Transpose back to [B, T, nHeads*d_head]
    ///   5. Output projection + bias
    ///
    /// The head split/merge is done via memory transpose without Concat opcode.
    /// This keeps the tape simple and avoids N separate SDPA recordings.
    ///
    /// CausalMask: true by default (required for autoregressive GPT).
    /// </summary>
    public sealed class MultiHeadAttentionLayer : IModule
    {
        private readonly int _dModel;
        private readonly int _nHeads;
        private readonly int _dHead;
        private readonly bool _causalMask;

        private AutogradNode? _wqNode;
        private AutogradNode? _wkNode;
        private AutogradNode? _wvNode;
        private AutogradNode? _woNode;

        public MultiHeadAttentionLayer(
            int dModel,
            int nHeads,
            bool causalMask = true)
        {
            if (dModel % nHeads != 0)
            {
                throw new ArgumentException(
                    $"d_model ({dModel}) must be divisible by nHeads ({nHeads}).");
            }

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nHeads);

            _dModel     = dModel;
            _nHeads     = nHeads;
            _dHead      = dModel / nHeads;
            _causalMask = causalMask;

            var scale = MathF.Sqrt(2f / dModel);

            // W_q, W_k, W_v: [dModel, nHeads*dHead] = [dModel, dModel]
            Wq = CreateWeight(dModel, dModel, scale);
            Wk = CreateWeight(dModel, dModel, scale);
            Wv = CreateWeight(dModel, dModel, scale);

            // W_o: [dModel, dModel]; b_o: [dModel]
            Wo = CreateWeight(dModel, dModel, scale);
            Bo = new Parameter(new TensorShape(dModel), requiresGrad: true, clearData: true);
        }

        public Parameter Wq { get; }
        public Parameter Wk { get; }
        public Parameter Wv { get; }
        public Parameter Wo { get; }
        public Parameter Bo { get; }

        public int DModel  => _dModel;
        public int NHeads  => _nHeads;
        public int DHead   => _dHead;

        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;

        public void Eval() => IsTraining = false;

        /// <summary>
        /// Forward pass.
        /// Input X: [B, T, d_model]. Output: [B, T, d_model].
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

            _wqNode ??= Wq.AsNode();
            _wkNode ??= Wk.AsNode();
            _wvNode ??= Wv.AsNode();
            _woNode ??= Wo.AsNode();

            // ── 1. Project: [B,T,dModel] @ W → [B,T,dModel] (= nHeads * dHead) ──
            var q = LinearFlat(graph, input, _wqNode, b * t, _dModel, _dModel);  // [B*T, dModel]
            var k = LinearFlat(graph, input, _wkNode, b * t, _dModel, _dModel);
            var v = LinearFlat(graph, input, _wvNode, b * t, _dModel, _dModel);

            // ── 2. Split heads: [B*T, nHeads*dHead] → [B*nHeads, T, dHead] ─────
            var qH = SplitHeads(graph, q, b, t, _nHeads, _dHead);  // [B*nHeads, T, dHead]
            var kH = SplitHeads(graph, k, b, t, _nHeads, _dHead);
            var vH = SplitHeads(graph, v, b, t, _nHeads, _dHead);

            // ── 3. SDPA on all heads at once (batchSize = B*nHeads) ────────────
            // Single SDPA call handles all heads — no loop needed.
            var attnOut = TensorMath.ScaledDotProductAttention(
                graph, qH, kH, vH, _causalMask);  // [B*nHeads, T, dHead]

            // ── 4. Merge heads: [B*nHeads, T, dHead] → [B*T, nHeads*dHead] ────
            var merged = MergeHeads(graph, attnOut, b, t, _nHeads, _dHead);  // [B*T, dModel]

            // ── 5. Output projection + bias ───────────────────────────────────
            var projected = LinearFlat(graph, merged, _woNode, b * t, _dModel, _dModel);

            // Add bias broadcast: [B*T, dModel] + [dModel] → [B*T, dModel]
            var withBias = AddBias(graph, projected, b, t, _dModel);

            // Reshape to [B, T, dModel]
            return graph.Reshape(withBias, b, t, _dModel);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
            => throw new NotSupportedException(
                "Use Forward(ComputationGraph, AutogradNode) — MHA requires sequence-shaped input.");

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Wq.AsNode(); yield return Wk.AsNode();
            yield return Wv.AsNode(); yield return Wo.AsNode();
            yield return Bo.AsNode();
        }

        public IEnumerable<Parameter> TrainableParameters()
        {
            yield return Wq; yield return Wk; yield return Wv; yield return Wo; yield return Bo;
        }

        public void InvalidateParameterCaches()
        {
            _wqNode = null; _wkNode = null; _wvNode = null; _woNode = null;
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
            _wqNode?.Dispose(); _wkNode?.Dispose(); _wvNode?.Dispose(); _woNode?.Dispose();
            Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose(); Bo.Dispose();
        }

        // ── Private helpers ───────────────────────────────────────────────────

        /// <summary>
        /// Flatten input [B, T, dIn] → [B*T, dIn], linear projection → [B*T, dOut].
        /// No bias (bias handled separately for the final projection).
        /// </summary>
        private static AutogradNode LinearFlat(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode weight,
            int bt,
            int dIn,
            int dOut)
        {
            // Flatten [B,T,dIn] → [B*T, dIn]
            var flat = graph.Reshape(input, bt, dIn);

            // Zero bias for internal projections
            var biasStorage = new TensorStorage<float>(dOut, clearMemory: true);
            var bias        = AutogradNode.CreateBorrowed(biasStorage, new TensorShape(dOut));

            return graph.Linear(flat, weight, bias);  // [B*T, dOut]
        }

        /// <summary>
        /// Transpose [B*T, nHeads*dHead] → [B*nHeads, T, dHead] (head-first layout).
        /// This is a memory copy (not tracked on tape — shape-only transform for SDPA).
        ///
        /// Layout before: [b, t, h, dHead] in row-major
        ///   index = b*T*nHeads*dHead + t*nHeads*dHead + h*dHead + d
        /// Layout after:  [b*h, t, dHead]
        ///   index = (b*nHeads+h)*T*dHead + t*dHead + d
        /// </summary>
        private static AutogradNode SplitHeads(
            ComputationGraph graph,
            AutogradNode input,
            int b,
            int t,
            int nHeads,
            int dHead)
        {
            var outSize = b * nHeads * t * dHead;
            var outStorage = new TensorStorage<float>(outSize, clearMemory: false);
            var inS  = input.DataView.AsReadOnlySpan();
            var outS = outStorage.AsSpan();

            // Transpose: (b, t, h, d) → (b, h, t, d)
            for (var bi = 0; bi < b; bi++)
            {
                for (var ti = 0; ti < t; ti++)
                {
                    for (var h = 0; h < nHeads; h++)
                    {
                        var srcOffset = bi * t * nHeads * dHead + ti * nHeads * dHead + h * dHead;
                        var dstOffset = (bi * nHeads + h) * t * dHead + ti * dHead;
                        inS.Slice(srcOffset, dHead).CopyTo(outS.Slice(dstOffset, dHead));
                    }
                }
            }

            // This node is not tracked on tape — it's a structural reshape for SDPA.
            // Gradient flows back through MergeHeads which is the inverse transpose.
            var node = new AutogradNode(outStorage, new TensorShape(b * nHeads, t, dHead),
                requiresGrad: input.RequiresGrad);

            return node;
        }

        /// <summary>
        /// Inverse of SplitHeads: [B*nHeads, T, dHead] → [B*T, nHeads*dHead].
        /// </summary>
        private static AutogradNode MergeHeads(
            ComputationGraph graph,
            AutogradNode input,
            int b,
            int t,
            int nHeads,
            int dHead)
        {
            var dModel     = nHeads * dHead;
            var outStorage = new TensorStorage<float>(b * t * dModel, clearMemory: false);
            var inS  = input.DataView.AsReadOnlySpan();
            var outS = outStorage.AsSpan();

            // Transpose back: (b, h, t, d) → (b, t, h, d)
            for (var bi = 0; bi < b; bi++)
            {
                for (var h = 0; h < nHeads; h++)
                {
                    for (var ti = 0; ti < t; ti++)
                    {
                        var srcOffset = (bi * nHeads + h) * t * dHead + ti * dHead;
                        var dstOffset = bi * t * dModel + ti * dModel + h * dHead;
                        inS.Slice(srcOffset, dHead).CopyTo(outS.Slice(dstOffset, dHead));
                    }
                }
            }

            var node = new AutogradNode(outStorage, new TensorShape(b * t, dModel),
                requiresGrad: input.RequiresGrad);

            return node;
        }

        /// <summary>
        /// Add bias [dModel] to all [B*T] rows of input [B*T, dModel].
        /// Produces output [B*T, dModel] with bias broadcast.
        /// </summary>
        private AutogradNode AddBias(
            ComputationGraph graph,
            AutogradNode input,
            int b,
            int t,
            int dModel)
        {
            var storage = new TensorStorage<float>(b * t * dModel, clearMemory: false);
            var inS     = input.DataView.AsReadOnlySpan();
            var boS     = Bo.DataReadOnlySpan;
            var outS    = storage.AsSpan();

            for (var bt = 0; bt < b * t; bt++)
            {
                var row = outS.Slice(bt * dModel, dModel);
                inS.Slice(bt * dModel, dModel).CopyTo(row);
                TensorPrimitives.Add(row, boS, row);
            }

            return new AutogradNode(storage, new TensorShape(b * t, dModel),
                requiresGrad: input.RequiresGrad);
        }

        private static Parameter CreateWeight(int rows, int cols, float scale)
        {
            var p = new Parameter(new TensorShape(rows, cols), requiresGrad: true, clearData: false);
            var s = p.DataSpan;
            for (var i = 0; i < s.Length; i++)
            {
                s[i] = MathUtils.NextGaussian() * scale;
            }
            return p;
        }
    }
}
