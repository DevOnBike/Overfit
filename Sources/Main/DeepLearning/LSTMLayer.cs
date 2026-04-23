// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class LstmLayer : IModule
    {
        public bool IsTraining { get; private set; } = true;

        private readonly LstmCell _cell;
        private readonly bool _returnSequences;

        public LstmLayer(int inputSize, int hiddenSize, bool returnSequences = false)
        {
            _cell = new LstmCell(inputSize, hiddenSize);
            _returnSequences = returnSequences;
        }

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

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var batch = input.Shape.D0;
            var seqLen = input.Shape.D1;
            var inputSize = input.Shape.D2;

            var (h, c) = _cell.ZeroState(batch);
            var allH = _returnSequences ? new AutogradNode[seqLen] : null;

            for (var t = 0; t < seqLen; t++)
            {
                var x_t = SliceTimestep(graph, input, t, seqLen, inputSize);
                (h, c) = _cell.Forward(graph, x_t, h, c);

                if (_returnSequences)
                {
                    allH[t] = h;
                }
            }

            if (_returnSequences)
            {
                return StackTimesteps(graph, allH, batch, seqLen);
            }

            return h;
        }

        public void ForwardInference(int batchSize, int seqLen, ReadOnlySpan<float> input, Span<float> output)
        {
            var inSize = _cell.InputSize;
            var hSize = _cell.HiddenSize;

            using var hBuf = new PooledBuffer<float>(batchSize * hSize);
            using var cBuf = new PooledBuffer<float>(batchSize * hSize);

            for (var t = 0; t < seqLen; t++)
            {
                var x_t = input.Slice(t * batchSize * inSize, batchSize * inSize);

                if (_returnSequences)
                {
                    var hNextOut = output.Slice(t * batchSize * hSize, batchSize * hSize);
                    _cell.ForwardInference(batchSize, x_t, hBuf.Span, cBuf.Span, hNextOut, cBuf.Span);
                    hNextOut.CopyTo(hBuf.Span);
                }
                else
                {
                    _cell.ForwardInference(batchSize, x_t, hBuf.Span, cBuf.Span, hBuf.Span, cBuf.Span);
                }
            }

            if (!_returnSequences)
            {
                hBuf.Span.CopyTo(output);
            }
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            throw new NotImplementedException("Use the overload providing batchSize and seqLen for LSTMLayer.");
        }

        private static AutogradNode SliceTimestep(ComputationGraph graph, AutogradNode input, int t, int seqLen, int inputSize)
        {
            var batch = input.Shape.D0;
            var res = new TensorStorage<float>(batch * inputSize, clearMemory: false);
            var dstS = res.AsSpan();
            var srcS = input.DataView.AsReadOnlySpan();

            for (var b = 0; b < batch; b++)
            {
                srcS.Slice(b * seqLen * inputSize + t * inputSize, inputSize).CopyTo(dstS.Slice(b * inputSize, inputSize));
            }

            var output = new AutogradNode(res, new TensorShape(batch, inputSize), input.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.TimestepSlice, output, input, null, t, seqLen, inputSize);
            }

            return output;
        }

        private static AutogradNode StackTimesteps(ComputationGraph graph, AutogradNode[] allH, int batch, int seqLen)
        {
            var hiddenSize = allH[0].Shape.D1;
            var res = new TensorStorage<float>(batch * seqLen * hiddenSize, clearMemory: false);
            var dstS = res.AsSpan();

            for (var t = 0; t < seqLen; t++)
            {
                var srcS = allH[t].DataView.AsReadOnlySpan();

                for (var b = 0; b < batch; b++)
                {
                    srcS.Slice(b * hiddenSize, hiddenSize).CopyTo(dstS.Slice(b * seqLen * hiddenSize + t * hiddenSize, hiddenSize));
                }
            }

            var requiresGrad = false;

            foreach (var h in allH)
            {
                if (h.RequiresGrad) { requiresGrad = true; break; }
            }

            var output = new AutogradNode(res, new TensorShape(batch, seqLen, hiddenSize), requiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.StackTimesteps, output, null, null, batch, seqLen, hiddenSize, 0, 0, allH);
            }

            return output;
        }

        public IEnumerable<AutogradNode> Parameters() => _cell.Parameters();
        public void InvalidateParameterCaches() => _cell.InvalidateParameterCaches();
        public void Save(BinaryWriter bw) => _cell.Save(bw);
        public void Load(BinaryReader br) => _cell.Load(br);
        public void Dispose() => _cell.Dispose();
        public void Save(string path) { }
        public void Load(string path) { }
    }
}