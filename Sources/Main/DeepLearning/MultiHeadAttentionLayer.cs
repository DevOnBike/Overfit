// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

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
    /// Multi-Head Attention — factored per-head implementation.
    ///
    /// Each head has its own small weight matrices [dModel, dHead].
    /// Everything is on tape — no off-tape transpositions (which blocked gradients).
    ///
    /// Per head h:
    ///   q_h = input @ Wq_h    [B*T, dModel] → [B*T, dHead]
    ///   k_h = input @ Wk_h
    ///   v_h = input @ Wv_h
    ///   q_h = Reshape → [B, T, dHead]   (pure shape, no data movement)
    ///   attn_h = SDPA(q_h, k_h, v_h)   [B, T, dHead]
    ///   proj_h = attn_h_flat @ Wo_h     [B*T, dModel]
    ///
    /// Output = sum_h(proj_h) + Bo, reshaped to [B, T, dModel].
    ///
    /// Parameter count: nHeads × 4 × dModel × dHead + dModel
    ///                = 4 × dModel² + dModel  (same as before)
    /// </summary>
    public sealed class MultiHeadAttentionLayer : IModule
    {
        private readonly int _dModel;
        private readonly int _nHeads;
        private readonly int _dHead;
        private readonly bool _causalMask;

        // Per-head weight matrices
        private readonly Parameter[] _wqHeads;
        private readonly Parameter[] _wkHeads;
        private readonly Parameter[] _wvHeads;
        private readonly Parameter[] _woHeads;

        // Cached nodes per head
        private readonly AutogradNode?[] _wqNodes;
        private readonly AutogradNode?[] _wkNodes;
        private readonly AutogradNode?[] _wvNodes;
        private readonly AutogradNode?[] _woNodes;

        public MultiHeadAttentionLayer(int dModel, int nHeads, bool causalMask = true)
        {
            if (dModel % nHeads != 0)
                throw new ArgumentException($"d_model ({dModel}) must be divisible by nHeads ({nHeads}).");

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nHeads);

            _dModel     = dModel;
            _nHeads     = nHeads;
            _dHead      = dModel / nHeads;
            _causalMask = causalMask;

            var scale = MathF.Sqrt(2f / dModel);

            _wqHeads = new Parameter[nHeads];
            _wkHeads = new Parameter[nHeads];
            _wvHeads = new Parameter[nHeads];
            _woHeads = new Parameter[nHeads];
            _wqNodes = new AutogradNode?[nHeads];
            _wkNodes = new AutogradNode?[nHeads];
            _wvNodes = new AutogradNode?[nHeads];
            _woNodes = new AutogradNode?[nHeads];

            for (var h = 0; h < nHeads; h++)
            {
                _wqHeads[h] = CreateWeight(dModel, _dHead, scale);
                _wkHeads[h] = CreateWeight(dModel, _dHead, scale);
                _wvHeads[h] = CreateWeight(dModel, _dHead, scale);
                _woHeads[h] = CreateWeight(_dHead, dModel, scale);
            }

            Bo = new Parameter(new TensorShape(dModel), requiresGrad: true, clearData: true);
        }

        // Compatibility properties — aggregate Wq/Wk/Wv/Wo for Save/Load
        public Parameter[] WqHeads => _wqHeads;
        public Parameter[] WkHeads => _wkHeads;
        public Parameter[] WvHeads => _wvHeads;
        public Parameter[] WoHeads => _woHeads;
        public Parameter Bo { get; }

        // Compatibility properties for existing tests and gradient checks
        public Parameter Wq => _wqHeads[0];
        public Parameter Wk => _wkHeads[0];
        public Parameter Wv => _wvHeads[0];
        public Parameter Wo => _woHeads[0];

        public int DModel => _dModel;
        public int NHeads => _nHeads;
        public int DHead  => _dHead;

        public bool IsTraining { get; private set; } = true;
        public void Train() => IsTraining = true;
        public void Eval()  => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var b      = input.Shape.D0;
            var t      = input.Shape.D1;
            var dModel = input.Shape.D2;
            var bt     = b * t;

            if (dModel != _dModel)
                throw new ArgumentException($"Expected d_model={_dModel}, got {dModel}.");

            // Flatten [B, T, dModel] → [B*T, dModel]
            var flat = graph.Reshape(input, bt, _dModel);

            // Process each head independently — all ops on tape
            AutogradNode? accum = null;

            for (var h = 0; h < _nHeads; h++)
            {
                _wqNodes[h] ??= _wqHeads[h].AsNode();
                _wkNodes[h] ??= _wkHeads[h].AsNode();
                _wvNodes[h] ??= _wvHeads[h].AsNode();
                _woNodes[h] ??= _woHeads[h].AsNode();

                // Q, K, V projections: [B*T, dModel] @ [dModel, dHead] → [B*T, dHead]
                var zeroBiasQ = ZeroBias(graph, _dHead);
                var zeroBiasK = ZeroBias(graph, _dHead);
                var zeroBiasV = ZeroBias(graph, _dHead);

                var qFlat = graph.Linear(flat, _wqNodes[h], zeroBiasQ);
                var kFlat = graph.Linear(flat, _wkNodes[h], zeroBiasK);
                var vFlat = graph.Linear(flat, _wvNodes[h], zeroBiasV);

                // Reshape to [B, T, dHead] — pure shape annotation, no data movement, ON TAPE
                var q3d = graph.Reshape(qFlat, b, t, _dHead);
                var k3d = graph.Reshape(kFlat, b, t, _dHead);
                var v3d = graph.Reshape(vFlat, b, t, _dHead);

                // SDPA: [B, T, dHead] → [B, T, dHead], ON TAPE
                var attn = TensorMath.ScaledDotProductAttention(graph, q3d, k3d, v3d, _causalMask);

                // Flatten: [B, T, dHead] → [B*T, dHead], ON TAPE
                var attnFlat = graph.Reshape(attn, bt, _dHead);

                // Output projection: [B*T, dHead] @ [dHead, dModel] → [B*T, dModel], ON TAPE
                var zeroBiasO = ZeroBias(graph, _dModel);
                var proj = graph.Linear(attnFlat, _woNodes[h], zeroBiasO);

                // Accumulate: output = sum of all head projections
                accum = accum == null ? proj : TensorMath.Add(graph, accum, proj);
            }

            // Add output bias [dModel], broadcast over B*T rows
            var boNode = Bo.AsNode();
            var withBias = AddBiasBroadcast(graph, accum!, boNode, bt, _dModel);

            // Reshape to [B, T, dModel]
            return graph.Reshape(withBias, b, t, _dModel);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
            => throw new NotSupportedException("Use Forward(ComputationGraph, AutogradNode).");

        public IEnumerable<AutogradNode> Parameters()
        {
            for (var h = 0; h < _nHeads; h++)
            {
                yield return _wqHeads[h].AsNode(); yield return _wkHeads[h].AsNode();
                yield return _wvHeads[h].AsNode(); yield return _woHeads[h].AsNode();
            }
            yield return Bo.AsNode();
        }

        public IEnumerable<Parameter> TrainableParameters()
        {
            for (var h = 0; h < _nHeads; h++)
            {
                yield return _wqHeads[h]; yield return _wkHeads[h];
                yield return _wvHeads[h]; yield return _woHeads[h];
            }
            yield return Bo;
        }

        public void InvalidateParameterCaches()
        {
            for (var h = 0; h < _nHeads; h++)
            {
                _wqNodes[h]?.Dispose(); _wqNodes[h] = null;
                _wkNodes[h]?.Dispose(); _wkNodes[h] = null;
                _wvNodes[h]?.Dispose(); _wvNodes[h] = null;
                _woNodes[h]?.Dispose(); _woNodes[h] = null;
            }
        }

        public void Save(BinaryWriter bw)
        {
            for (var h = 0; h < _nHeads; h++)
            {
                _wqHeads[h].Save(bw); _wkHeads[h].Save(bw);
                _wvHeads[h].Save(bw); _woHeads[h].Save(bw);
            }
            Bo.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            for (var h = 0; h < _nHeads; h++)
            {
                _wqHeads[h].Load(br); _wkHeads[h].Load(br);
                _wvHeads[h].Load(br); _woHeads[h].Load(br);
            }
            Bo.Load(br);
        }

        public void Dispose()
        {
            InvalidateParameterCaches();
            for (var h = 0; h < _nHeads; h++)
            {
                _wqHeads[h].Dispose(); _wkHeads[h].Dispose();
                _wvHeads[h].Dispose(); _woHeads[h].Dispose();
            }
            Bo.Dispose();
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        private static AutogradNode ZeroBias(ComputationGraph graph, int size)
        {
            var storage = new TensorStorage<float>(size, clearMemory: true);
            return AutogradNode.CreateBorrowed(storage, new TensorShape(size));
        }

        private static AutogradNode AddBiasBroadcast(
            ComputationGraph graph, AutogradNode input, AutogradNode bias, int bt, int dModel)
        {
            // graph.AddBias broadcasts [dModel] over [B*T, dModel] ON TAPE.
            // Backward: dInput += dOutput (element-wise), dBias[j] += sum_i(dOutput[i,j]).
            return graph.AddBias(input, bias);
        }

        private static Parameter CreateWeight(int rows, int cols, float scale)
        {
            var p = new Parameter(new TensorShape(rows, cols), requiresGrad: true, clearData: false);
            var s = p.DataSpan;
            for (var i = 0; i < s.Length; i++) s[i] = MathUtils.NextGaussian() * scale;
            return p;
        }
    }
}
