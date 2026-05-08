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

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Multi-Head Attention — factored per-head implementation.
    ///
    /// Each head has its own small weight matrices [dModel, dHead].
    /// Everything is on tape — no off-tape transpositions.
    ///
    /// Per head:
    /// q_h = input @ Wq_h + Bq_h [B*T, dModel] -> [B*T, dHead]
    /// k_h = input @ Wk_h + Bk_h
    /// v_h = input @ Wv_h + Bv_h
    /// q_h = reshape -> [B, T, dHead]
    /// attn_h = SDPA(q_h, k_h, v_h) [B, T, dHead]
    /// proj_h = attn_h_flat @ Wo_h [B*T, dModel]
    ///
    /// Output = sum_h(proj_h) + Bo, reshaped to [B, T, dModel].
    /// </summary>
    public sealed class MultiHeadAttentionLayer : IModule
    {
        private readonly int _dModel;
        private readonly int _nHeads;
        private readonly int _dHead;
        private readonly bool _causalMask;

        private readonly Parameter[] _wqHeads;
        private readonly Parameter[] _wkHeads;
        private readonly Parameter[] _wvHeads;
        private readonly Parameter[] _woHeads;

        // Per-head Q/K/V biases. Zero-initialized for training-from-scratch.
        // Populated from GPT-2 c_attn.bias when loading converted GPT-2 checkpoints.
        private readonly Parameter[] _bqHeads;
        private readonly Parameter[] _bkHeads;
        private readonly Parameter[] _bvHeads;

        private readonly AutogradNode?[] _wqNodes;
        private readonly AutogradNode?[] _wkNodes;
        private readonly AutogradNode?[] _wvNodes;
        private readonly AutogradNode?[] _woNodes;
        private readonly AutogradNode?[] _bqNodes;
        private readonly AutogradNode?[] _bkNodes;
        private readonly AutogradNode?[] _bvNodes;

        public MultiHeadAttentionLayer(
            int dModel,
            int nHeads,
            bool causalMask = true)
        {
            if (dModel % nHeads != 0)
            {
                throw new ArgumentException($"d_model ({dModel}) must be divisible by nHeads ({nHeads}).");
            }

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nHeads);

            _dModel = dModel;
            _nHeads = nHeads;
            _dHead = dModel / nHeads;
            _causalMask = causalMask;

            var scale = MathF.Sqrt(2f / dModel);

            _wqHeads = new Parameter[nHeads];
            _wkHeads = new Parameter[nHeads];
            _wvHeads = new Parameter[nHeads];
            _woHeads = new Parameter[nHeads];

            _bqHeads = new Parameter[nHeads];
            _bkHeads = new Parameter[nHeads];
            _bvHeads = new Parameter[nHeads];

            _wqNodes = new AutogradNode?[nHeads];
            _wkNodes = new AutogradNode?[nHeads];
            _wvNodes = new AutogradNode?[nHeads];
            _woNodes = new AutogradNode?[nHeads];
            _bqNodes = new AutogradNode?[nHeads];
            _bkNodes = new AutogradNode?[nHeads];
            _bvNodes = new AutogradNode?[nHeads];

            for (var h = 0; h < nHeads; h++)
            {
                _wqHeads[h] = CreateWeight(dModel, _dHead, scale);
                _wkHeads[h] = CreateWeight(dModel, _dHead, scale);
                _wvHeads[h] = CreateWeight(dModel, _dHead, scale);
                _woHeads[h] = CreateWeight(_dHead, dModel, scale);

                _bqHeads[h] = CreateBias(_dHead);
                _bkHeads[h] = CreateBias(_dHead);
                _bvHeads[h] = CreateBias(_dHead);
            }

            Bo = new Parameter(
            new TensorShape(dModel),
            requiresGrad: true,
            clearData: true);
        }

        public Parameter[] WqHeads => _wqHeads;
        public Parameter[] WkHeads => _wkHeads;
        public Parameter[] WvHeads => _wvHeads;
        public Parameter[] WoHeads => _woHeads;

        public Parameter[] BqHeads => _bqHeads;
        public Parameter[] BkHeads => _bkHeads;
        public Parameter[] BvHeads => _bvHeads;

        public Parameter Bo { get; }

        // Compatibility properties for existing tests and gradient checks.
        public Parameter Wq => _wqHeads[0];
        public Parameter Wk => _wkHeads[0];
        public Parameter Wv => _wvHeads[0];
        public Parameter Wo => _woHeads[0];
        public Parameter Bq => _bqHeads[0];
        public Parameter Bk => _bkHeads[0];
        public Parameter Bv => _bvHeads[0];

        public int DModel => _dModel;
        public int NHeads => _nHeads;
        public int DHead => _dHead;

        public bool IsTraining { get; private set; } = true;

        public void Train()
        {
            IsTraining = true;
        }

        public void Eval()
        {
            IsTraining = false;
        }

        public AutogradNode Forward(
            ComputationGraph graph,
            AutogradNode input)
        {
            if (graph is null)
            {
                throw new ArgumentNullException(nameof(graph));
            }

            if (input is null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            var batchSize = input.Shape.D0;
            var seqLen = input.Shape.D1;
            var dModel = input.Shape.D2;
            var batchTime = batchSize * seqLen;

            if (dModel != _dModel)
            {
                throw new ArgumentException($"Expected d_model={_dModel}, got {dModel}.");
            }

            // Flatten [B, T, dModel] -> [B*T, dModel].
            var flat = graph.Reshape(
            input,
            batchTime,
            _dModel);

            AutogradNode? accum = null;

            for (var h = 0; h < _nHeads; h++)
            {
                _wqNodes[h] ??= _wqHeads[h].AsNode();
                _wkNodes[h] ??= _wkHeads[h].AsNode();
                _wvNodes[h] ??= _wvHeads[h].AsNode();
                _woNodes[h] ??= _woHeads[h].AsNode();

                _bqNodes[h] ??= _bqHeads[h].AsNode();
                _bkNodes[h] ??= _bkHeads[h].AsNode();
                _bvNodes[h] ??= _bvHeads[h].AsNode();

                var qFlat = graph.Linear(flat, _wqNodes[h]!, _bqNodes[h]!);
                var kFlat = graph.Linear(flat, _wkNodes[h]!, _bkNodes[h]!);
                var vFlat = graph.Linear(flat, _wvNodes[h]!, _bvNodes[h]!);

                var q3d = graph.Reshape(
                qFlat,
                batchSize,
                seqLen,
                _dHead);

                var k3d = graph.Reshape(
                kFlat,
                batchSize,
                seqLen,
                _dHead);

                var v3d = graph.Reshape(
                vFlat,
                batchSize,
                seqLen,
                _dHead);

                var attn = TensorMath.ScaledDotProductAttention(
                graph,
                q3d,
                k3d,
                v3d,
                _causalMask);

                var attnFlat = graph.Reshape(
                attn,
                batchTime,
                _dHead);

                var zeroBiasO = ZeroBias(graph, _dModel);

                var proj = graph.Linear(
                attnFlat,
                _woNodes[h]!,
                zeroBiasO);

                accum = accum == null
                    ? proj
                    : TensorMath.Add(graph, accum, proj);
            }

            var boNode = Bo.AsNode();

            var withBias = AddBiasBroadcast(
            graph,
            accum!,
            boNode,
            batchTime,
            _dModel);

            return graph.Reshape(
            withBias,
            batchSize,
            seqLen,
            _dModel);
        }

        public void ForwardInference(
            ReadOnlySpan<float> input,
            Span<float> output)
        {
            throw new NotSupportedException("Use Forward(ComputationGraph, AutogradNode).");
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            for (var h = 0; h < _nHeads; h++)
            {
                yield return _wqHeads[h].AsNode();
                yield return _bqHeads[h].AsNode();

                yield return _wkHeads[h].AsNode();
                yield return _bkHeads[h].AsNode();

                yield return _wvHeads[h].AsNode();
                yield return _bvHeads[h].AsNode();

                yield return _woHeads[h].AsNode();
            }

            yield return Bo.AsNode();
        }

        public IEnumerable<Parameter> TrainableParameters()
        {
            for (var h = 0; h < _nHeads; h++)
            {
                yield return _wqHeads[h];
                yield return _bqHeads[h];

                yield return _wkHeads[h];
                yield return _bkHeads[h];

                yield return _wvHeads[h];
                yield return _bvHeads[h];

                yield return _woHeads[h];
            }

            yield return Bo;
        }

        public void InvalidateParameterCaches()
        {
            for (var h = 0; h < _nHeads; h++)
            {
                _wqNodes[h]?.Dispose();
                _wqNodes[h] = null;

                _wkNodes[h]?.Dispose();
                _wkNodes[h] = null;

                _wvNodes[h]?.Dispose();
                _wvNodes[h] = null;

                _woNodes[h]?.Dispose();
                _woNodes[h] = null;

                _bqNodes[h]?.Dispose();
                _bqNodes[h] = null;

                _bkNodes[h]?.Dispose();
                _bkNodes[h] = null;

                _bvNodes[h]?.Dispose();
                _bvNodes[h] = null;
            }
        }

        public void Save(BinaryWriter writer)
        {
            if (writer is null)
            {
                throw new ArgumentNullException(nameof(writer));
            }

            for (var h = 0; h < _nHeads; h++)
            {
                _wqHeads[h].Save(writer);
                _bqHeads[h].Save(writer);

                _wkHeads[h].Save(writer);
                _bkHeads[h].Save(writer);

                _wvHeads[h].Save(writer);
                _bvHeads[h].Save(writer);

                _woHeads[h].Save(writer);
            }

            Bo.Save(writer);
        }

        public void Load(BinaryReader reader)
        {
            if (reader is null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            _wqHeads[0].Load(reader);

            if (IsNewQkvBiasCheckpointFormat(reader))
            {
                LoadNewFormatAfterFirstWq(reader);
            }
            else
            {
                LoadLegacyFormatAfterFirstWq(reader);
            }
        }

        public void Dispose()
        {
            InvalidateParameterCaches();

            for (var h = 0; h < _nHeads; h++)
            {
                _wqHeads[h].Dispose();
                _bqHeads[h].Dispose();

                _wkHeads[h].Dispose();
                _bkHeads[h].Dispose();

                _wvHeads[h].Dispose();
                _bvHeads[h].Dispose();

                _woHeads[h].Dispose();
            }

            Bo.Dispose();
        }

        private bool IsNewQkvBiasCheckpointFormat(BinaryReader reader)
        {
            var stream = reader.BaseStream;

            if (!stream.CanSeek)
            {
                throw new NotSupportedException(
                "MultiHeadAttentionLayer.Load requires a seekable stream to detect legacy checkpoints after Q/K/V bias support was added.");
            }

            var position = stream.Position;
            var nextParameterLength = reader.ReadInt32();
            stream.Position = position;

            if (nextParameterLength == _dHead)
            {
                return true;
            }

            if (nextParameterLength == _dModel * _dHead)
            {
                return false;
            }

            throw new InvalidDataException(
            $"Unexpected parameter length {nextParameterLength} after Wq. " +
            $"Expected {_dHead} for new Q/K/V-bias checkpoints or {_dModel * _dHead} for legacy checkpoints.");
        }

        private void LoadNewFormatAfterFirstWq(BinaryReader reader)
        {
            _bqHeads[0].Load(reader);
            _wkHeads[0].Load(reader);
            _bkHeads[0].Load(reader);
            _wvHeads[0].Load(reader);
            _bvHeads[0].Load(reader);
            _woHeads[0].Load(reader);

            for (var h = 1; h < _nHeads; h++)
            {
                _wqHeads[h].Load(reader);
                _bqHeads[h].Load(reader);

                _wkHeads[h].Load(reader);
                _bkHeads[h].Load(reader);

                _wvHeads[h].Load(reader);
                _bvHeads[h].Load(reader);

                _woHeads[h].Load(reader);
            }

            Bo.Load(reader);
        }

        private void LoadLegacyFormatAfterFirstWq(BinaryReader reader)
        {
            ClearQkvBiases(0);

            _wkHeads[0].Load(reader);
            _wvHeads[0].Load(reader);
            _woHeads[0].Load(reader);

            for (var h = 1; h < _nHeads; h++)
            {
                _wqHeads[h].Load(reader);
                ClearQkvBiases(h);

                _wkHeads[h].Load(reader);
                _wvHeads[h].Load(reader);
                _woHeads[h].Load(reader);
            }

            Bo.Load(reader);
        }

        private void ClearQkvBiases(int head)
        {
            _bqHeads[head].DataSpan.Clear();
            _bkHeads[head].DataSpan.Clear();
            _bvHeads[head].DataSpan.Clear();
        }

        private static AutogradNode ZeroBias(
            ComputationGraph graph,
            int size)
        {
            // This is a graph-owned zero tensor.
            //
            // The previous implementation allocated a TensorStorage locally and
            // returned AutogradNode.CreateBorrowed(...), which incorrectly tagged
            // the node as ExternalBorrowed. Since Reset() only disposes
            // GraphTemporary/GraphAuxiliary nodes, that was the wrong ownership
            // classification for a per-forward temporary bias.
            return graph.CreateAuxiliary(
            new TensorShape(size),
            clearMemory: true);
        }

        private static AutogradNode AddBiasBroadcast(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode bias,
            int batchTime,
            int dModel)
        {
            _ = batchTime;
            _ = dModel;
            return graph.AddBias(input, bias);
        }

        private static Parameter CreateWeight(
            int rows,
            int cols,
            float scale)
        {
            var parameter = new Parameter(
            new TensorShape(rows, cols),
            requiresGrad: true,
            clearData: false);

            var data = parameter.DataSpan;

            for (var i = 0; i < data.Length; i++)
            {
                data[i] = MathUtils.NextGaussian() * scale;
            }

            return parameter;
        }

        private static Parameter CreateBias(int size)
        {
            return new Parameter(
            new TensorShape(size),
            requiresGrad: true,
            clearData: true);
        }
    }
}
