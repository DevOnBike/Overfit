// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class LSTMLayer : IModule
    {
        public bool IsTraining { get; private set; } = true;

        private readonly LSTMCell _cell;
        private readonly bool _returnSequences;

        public LSTMLayer(int inputSize, int hiddenSize, bool returnSequences = false)
        {
            _cell = new LSTMCell(inputSize, hiddenSize); 
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
            var batch = input.DataView.GetDim(0);
            var seqLen = input.DataView.GetDim(1);
            var inputSize = input.DataView.GetDim(2);

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

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            // Adapter dla interfejsu IModule (Batch = 1).
            // Automatycznie wylicza długość sekwencji z przesłanego spana.
            var inputSize = _cell.InputSize;
            var seqLen = input.Length / inputSize;

            // Wywołujemy istniejącą pętlę BPTT w trybie inferencji
            ForwardInference(1, seqLen, input, output);
        }

        public void ForwardInference(int batchSize, int seqLen, ReadOnlySpan<float> input, Span<float> output)
        {
            var inputSize = _cell.InputSize;
            var hiddenSize = _cell.HiddenSize;
            var gatesLen = batchSize * 4 * hiddenSize;

            using var hBuf = new PooledBuffer<float>(batchSize * hiddenSize);
            using var cBuf = new PooledBuffer<float>(batchSize * hiddenSize);
            using var xtBuf = new PooledBuffer<float>(batchSize * inputSize);
            using var gatesBuf = new PooledBuffer<float>(gatesLen);
            using var uhBuf = new PooledBuffer<float>(gatesLen);

            var h = hBuf.Span;
            var c = cBuf.Span;
            var xt = xtBuf.Span;
            var gates = gatesBuf.Span;
            var uh = uhBuf.Span;

            h.Clear();
            c.Clear();

            for (var t = 0; t < seqLen; t++)
            {
                for (var b = 0; b < batchSize; b++)
                {
                    input.Slice(b * seqLen * inputSize + t * inputSize, inputSize).CopyTo(xt.Slice(b * inputSize, inputSize));
                }

                _cell.ForwardInference(batchSize, xt, h, c, gates, uh);

                if (_returnSequences)
                {
                    for (var b = 0; b < batchSize; b++)
                    {
                        h.Slice(b * hiddenSize, hiddenSize).CopyTo(output.Slice(b * seqLen * hiddenSize + t * hiddenSize, hiddenSize));
                    }
                }
            }

            if (!_returnSequences)
            {
                h.CopyTo(output);
            }
        }

        private static AutogradNode ExtractTimestep(ComputationGraph graph, AutogradNode input, int t, int batch, int seqLen, int inputSize)
        {
            var res = new FastTensor<float>(batch, inputSize, clearMemory: false);
            var srcS = input.DataView.AsReadOnlySpan();
            var dstS = res.GetView().AsSpan();

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
            var hiddenSize = allH[0].DataView.GetDim(1);
            var res = new FastTensor<float>(batch, seqLen, hiddenSize, clearMemory: false);
            var dstS = res.GetView().AsSpan();

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

            var output = new AutogradNode(res, requiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.StackTimesteps, output, null, null, batch, seqLen, hiddenSize, nodeContext: allH);
            }

            return output;
        }
    }
}