// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    ///     LSTM layer — wraps LSTMCell and iterates over sequence timesteps.
    ///     Includes zero-allocation path via ForwardInference.
    /// </summary>
    public sealed class LSTMLayer : IModule
    {
        private readonly LSTMCell _cell;
        private readonly bool _returnSequences;

        public LSTMLayer(int inputSize, int hiddenSize, bool returnSequences = false)
        {
            _cell = new LSTMCell(inputSize, hiddenSize);
            _returnSequences = returnSequences;
        }

        public bool IsTraining { get; private set; } = true;

        public void Train()
        {
            IsTraining = true;
            _cell.Train();
        }

        public void Eval()
        {
            IsTraining = false;
            _cell.Eval();
        }

        // ---------------------------------------------------------------------------
        // Forward — Training Path
        // ---------------------------------------------------------------------------

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var batch = input.Data.GetDim(0);
            var seqLen = input.Data.GetDim(1);
            var inputSize = input.Data.GetDim(2);

            var (h, c) = _cell.ZeroState(batch);

            if (_returnSequences)
            {
                var allH = new AutogradNode[seqLen];

                for (var t = 0; t < seqLen; t++)
                {
                    var xt = ExtractTimestep(graph, input, t, batch, seqLen, inputSize);
                    (h, c) = _cell.Forward(graph, xt, h, c);
                    allH[t] = h;
                }

                return StackTimesteps(graph, allH, batch, seqLen);
            }
            for (var t = 0; t < seqLen; t++)
            {
                var xt = ExtractTimestep(graph, input, t, batch, seqLen, inputSize);
                (h, c) = _cell.Forward(graph, xt, h, c);
            }

            return h;
        }

        // ---------------------------------------------------------------------------
        // IModule
        // ---------------------------------------------------------------------------

        public IEnumerable<AutogradNode> Parameters()
        {
            return _cell.Parameters();
        }
        public void Save(BinaryWriter bw)
        {
            _cell.Save(bw);
        }
        public void Load(BinaryReader br)
        {
            _cell.Load(br);
        }
        public void Dispose()
        {
            _cell.Dispose();
        }

        // ---------------------------------------------------------------------------
        // Forward — Zero-Allocation Inference Path
        // ---------------------------------------------------------------------------

        /// <summary>
        ///     Zero-allocation forward pass for production inference.
        ///     input:  [batchSize, seqLen, inputSize]
        ///     output: [batchSize, seqLen, hiddenSize] OR [batchSize, hiddenSize]
        /// </summary>
        public void ForwardInference(int batchSize, int seqLen, ReadOnlySpan<float> input, Span<float> output)
        {
            var inputSize = _cell.InputSize;
            var hiddenSize = _cell.HiddenSize;

            var hArr = ArrayPool<float>.Shared.Rent(batchSize * hiddenSize);
            var cArr = ArrayPool<float>.Shared.Rent(batchSize * hiddenSize);
            var xtArr = ArrayPool<float>.Shared.Rent(batchSize * inputSize);

            try
            {
                var h = hArr.AsSpan(0, batchSize * hiddenSize);
                var c = cArr.AsSpan(0, batchSize * hiddenSize);
                var xt = xtArr.AsSpan(0, batchSize * inputSize);

                h.Clear();
                c.Clear();

                for (var t = 0; t < seqLen; t++)
                {
                    // Gather timestep
                    for (var b = 0; b < batchSize; b++)
                    {
                        input.Slice(b * seqLen * inputSize + t * inputSize, inputSize)
                            .CopyTo(xt.Slice(b * inputSize, inputSize));
                    }

                    _cell.ForwardInference(batchSize, xt, h, c);

                    if (_returnSequences)
                    {
                        // Scatter hidden state
                        for (var b = 0; b < batchSize; b++)
                        {
                            h.Slice(b * hiddenSize, hiddenSize)
                                .CopyTo(output.Slice(b * seqLen * hiddenSize + t * hiddenSize, hiddenSize));
                        }
                    }
                }

                if (!_returnSequences)
                {
                    // Copy final hidden state to output
                    h.CopyTo(output);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(hArr);
                ArrayPool<float>.Shared.Return(cArr);
                ArrayPool<float>.Shared.Return(xtArr);
            }
        }

        private static AutogradNode ExtractTimestep(ComputationGraph graph, AutogradNode input, int t, int batch, int seqLen, int inputSize)
        {
            var res = new FastTensor<float>(false, batch, inputSize);
            var srcS = input.Data.AsReadOnlySpan();
            var dstS = res.AsSpan();

            for (var b = 0; b < batch; b++)
            {
                srcS.Slice(b * seqLen * inputSize + t * inputSize, inputSize).CopyTo(dstS.Slice(b * inputSize, inputSize));
            }

            var output = new AutogradNode(res, input.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.TimestepSlice, output, input, null, t, seqLen, inputSize);
            }
            return output;
        }

        private static AutogradNode StackTimesteps(ComputationGraph graph, AutogradNode[] allH, int batch, int seqLen)
        {
            var hiddenSize = allH[0].Data.GetDim(1);
            var res = new FastTensor<float>(false, batch, seqLen, hiddenSize);
            var dstS = res.AsSpan();

            for (var t = 0; t < seqLen; t++)
            {
                var srcS = allH[t].Data.AsReadOnlySpan();

                for (var b = 0; b < batch; b++)
                {
                    srcS.Slice(b * hiddenSize, hiddenSize).CopyTo(dstS.Slice(b * seqLen * hiddenSize + t * hiddenSize, hiddenSize));
                }
            }

            var requiresGrad = false;

            foreach (var h in allH)
            {
                if (h.RequiresGrad)
                {
                    requiresGrad = true;
                    break;
                }
            }

            var output = new AutogradNode(res, requiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.StackTimesteps, output, null, null, batch, seqLen, hiddenSize, nodeContext: allH);
            }

            return output;
        }
    }
}