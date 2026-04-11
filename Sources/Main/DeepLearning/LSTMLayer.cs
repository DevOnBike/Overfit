// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// LSTM layer — wraps LSTMCell and iterates over sequence timesteps.
    ///
    /// Input:  [batch, seqLen, inputSize]
    /// Output: returnSequences=true  → [batch, seqLen, hiddenSize]  (decoder use)
    ///         returnSequences=false → [batch, hiddenSize]           (encoder use — last h only)
    ///
    /// Usage in autoencoder:
    ///   Encoder: LSTMLayer(12, 64, returnSequences:true)  → LSTMLayer(64, 32, returnSequences:false)
    ///   Decoder: LSTMLayer(32, 64, returnSequences:true)  → LSTMLayer(64, 12, returnSequences:true)
    /// </summary>
    public sealed class LSTMLayer : IModule
    {
        private readonly LSTMCell _cell;
        private readonly bool _returnSequences;

        public bool IsTraining { get; private set; } = true;

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

        // ---------------------------------------------------------------------------
        // Forward
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Forward pass over the full sequence.
        /// input shape: [batch, seqLen, inputSize]
        /// </summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var batch = input.Data.GetDim(0);
            var seqLen = input.Data.GetDim(1);
            var inputSize = input.Data.GetDim(2);

            var (h, c) = _cell.ZeroState(batch);

            if (_returnSequences)
            {
                // Accumulate all hidden states → [batch, seqLen, hiddenSize]
                var allH = new AutogradNode[seqLen];

                for (var t = 0; t < seqLen; t++)
                {
                    var xt = ExtractTimestep(graph, input, t, batch, seqLen, inputSize);
                    (h, c) = _cell.Forward(graph, xt, h, c);
                    allH[t] = h;
                }

                return StackTimesteps(graph, allH, batch, seqLen);
            }
            else
            {
                // Return only last hidden state → [batch, hiddenSize]
                for (var t = 0; t < seqLen; t++)
                {
                    var xt = ExtractTimestep(graph, input, t, batch, seqLen, inputSize);
                    (h, c) = _cell.Forward(graph, xt, h, c);
                }

                return h;
            }
        }

        // ---------------------------------------------------------------------------
        // Extract single timestep [batch, inputSize] from [batch, seqLen, inputSize]
        // ---------------------------------------------------------------------------

        private static AutogradNode ExtractTimestep(
            ComputationGraph graph,
            AutogradNode input,
            int t,
            int batch,
            int seqLen,
            int inputSize)
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

        // ---------------------------------------------------------------------------
        // Stack all hidden states → [batch, seqLen, hiddenSize]
        // ---------------------------------------------------------------------------

        private static AutogradNode StackTimesteps(
            ComputationGraph graph,
            AutogradNode[] allH,
            int batch,
            int seqLen)
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
                if (h.RequiresGrad) { requiresGrad = true; break; }
            }

            var output = new AutogradNode(res, requiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.StackTimesteps, output, null, null, batch, seqLen, hiddenSize, nodeContext: allH);
            }

            return output;
        }

        // ---------------------------------------------------------------------------
        // IModule
        // ---------------------------------------------------------------------------

        public IEnumerable<AutogradNode> Parameters() => _cell.Parameters();

        public void Save(BinaryWriter bw) => _cell.Save(bw);

        public void Load(BinaryReader br) => _cell.Load(br);

        public void Dispose() => _cell.Dispose();
    }
}