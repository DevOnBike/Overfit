// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class LstmCell : IModule
    {
        public int InputSize { get; }
        public int HiddenSize { get; }
        public AutogradNode W { get; }
        public AutogradNode U { get; }
        public AutogradNode B { get; }
        public bool IsTraining { get; private set; } = true;

        public LstmCell(int inputSize, int hiddenSize)
        {
            InputSize = inputSize; HiddenSize = hiddenSize;
            var limit = MathF.Sqrt(6f / (inputSize + hiddenSize));

            W = new AutogradNode(new FastTensor<float>(inputSize, 4 * hiddenSize, clearMemory: false), true);
            U = new AutogradNode(new FastTensor<float>(hiddenSize, 4 * hiddenSize, clearMemory: false), true);
            B = new AutogradNode(new FastTensor<float>(4 * hiddenSize, clearMemory: false), true);

            InitUniform(W.DataView.AsSpan(), limit);
            InitUniform(U.DataView.AsSpan(), limit);

            var bSpan = B.DataView.AsSpan();
            bSpan.Fill(0f);
            bSpan.Slice(0, hiddenSize).Fill(1f); // Forget gate bias
        }

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            // Adapter dla interfejsu IModule (pojedynczy krok czasowy, Batch = 1).
            var hSize = HiddenSize;

            using var cBuf = new PooledBuffer<float>(hSize);
            using var gatesBuf = new PooledBuffer<float>(4 * hSize);
            using var uhBuf = new PooledBuffer<float>(4 * hSize);

            var c = cBuf.Span;
            var gates = gatesBuf.Span;
            var uh = uhBuf.Span;

            c.Clear();
            output.Clear(); // W LSTMie wynik wyjściowy (H) stanowi bezpośrednio nasze 'output'

            ForwardInference(1, input, output, c, gates, uh);
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            throw new InvalidOperationException("Use Forward(graph, x, h, c) instead.");
        }

        public IEnumerable<AutogradNode> Parameters() { yield return W; yield return U; yield return B; }

        public void Save(BinaryWriter bw)
        {
            foreach (var v in W.DataView.AsSpan())
            {
                bw.Write(v);
            }
            foreach (var v in U.DataView.AsSpan())
            {
                bw.Write(v);
            }
            foreach (var v in B.DataView.AsSpan())
            {
                bw.Write(v);
            }
        }

        public void Load(BinaryReader br)
        {
            var wSpan = W.DataView.AsSpan();
            for (var i = 0; i < wSpan.Length; i++)
            {
                wSpan[i] = br.ReadSingle();
            }

            var uSpan = U.DataView.AsSpan();
            for (var i = 0; i < uSpan.Length; i++)
            {
                uSpan[i] = br.ReadSingle();
            }

            var bSpan = B.DataView.AsSpan();
            for (var i = 0; i < bSpan.Length; i++)
            {
                bSpan[i] = br.ReadSingle();
            }
        }

        public void Dispose()
        {
            W?.Dispose();
            U?.Dispose();
            B?.Dispose();
        }

        public void ForwardInference(int batchSize, ReadOnlySpan<float> x, Span<float> h, Span<float> c, Span<float> gates, Span<float> uh)
        {
            var hSize = HiddenSize;
            var wSpan = W.DataView.AsReadOnlySpan();
            var uSpan = U.DataView.AsReadOnlySpan();
            var bSpan = B.DataView.AsReadOnlySpan();

            gates.Clear();
            uh.Clear();

            for (var b = 0; b < batchSize; b++)
            {
                var bGates = gates.Slice(b * 4 * hSize, 4 * hSize);
                var bX = x.Slice(b * InputSize, InputSize);

                for (var i = 0; i < InputSize; i++)
                {
                    if (bX[i] != 0f)
                    {
                        TensorPrimitives.MultiplyAdd(wSpan.Slice(i * 4 * hSize, 4 * hSize), bX[i], bGates, bGates);
                    }
                }
                TensorPrimitives.Add(bGates, bSpan, bGates);

                var bUh = uh.Slice(b * 4 * hSize, 4 * hSize);
                var bH = h.Slice(b * hSize, hSize);

                for (var i = 0; i < hSize; i++)
                {
                    if (bH[i] != 0f)
                    {
                        TensorPrimitives.MultiplyAdd(uSpan.Slice(i * 4 * hSize, 4 * hSize), bH[i], bUh, bUh);
                    }
                }
                
                TensorPrimitives.Add(bGates, bUh, bGates);

                var bC = c.Slice(b * hSize, hSize);
                var gF = bGates.Slice(0, hSize);
                var gI = bGates.Slice(hSize, hSize);
                var gG = bGates.Slice(2 * hSize, hSize);
                var gO = bGates.Slice(3 * hSize, hSize);

                TensorPrimitives.Sigmoid(gF, gF);
                TensorPrimitives.Sigmoid(gI, gI);
                TensorPrimitives.Tanh(gG, gG);
                TensorPrimitives.Sigmoid(gO, gO);

                TensorPrimitives.Multiply(gF, bC, bC);
                TensorPrimitives.MultiplyAdd(gI, gG, bC, bC);
                TensorPrimitives.Tanh(bC, gG);
                TensorPrimitives.Multiply(gO, gG, bH);
            }
        }

        public (AutogradNode h, AutogradNode c) Forward(ComputationGraph graph, AutogradNode x, AutogradNode h, AutogradNode c)
        {
            return TensorMath.FusedLSTMStep(graph, x, h, c, W, U, B);
        }

        public (AutogradNode h0, AutogradNode c0) ZeroState(int batchSize)
        {
            var h0 = new AutogradNode(new FastTensor<float>(batchSize, HiddenSize), false);
            var c0 = new AutogradNode(new FastTensor<float>(batchSize, HiddenSize), false);

            return (h0, c0);
        }

        private static void InitUniform(Span<float> span, float limit)
        {
            var rng = Random.Shared;
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = (float)(rng.NextDouble() * 2.0 * limit - limit);
            }
        }

        public void InvalidateParameterCaches()
        {
        }
    }
}