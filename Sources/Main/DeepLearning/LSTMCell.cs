// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

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

            W = new AutogradNode(new TensorStorage<float>(inputSize * 4 * hiddenSize, clearMemory: false), new TensorShape(inputSize, 4 * hiddenSize), true);
            U = new AutogradNode(new TensorStorage<float>(hiddenSize * 4 * hiddenSize, clearMemory: false), new TensorShape(hiddenSize, 4 * hiddenSize), true);
            B = new AutogradNode(new TensorStorage<float>(4 * hiddenSize, clearMemory: true), new TensorShape(4 * hiddenSize), true);

            InitUniform(W.DataView.AsSpan(), limit);
            InitUniform(U.DataView.AsSpan(), limit);
        }

        public void Train()
        {
            IsTraining = true;
        }
        public void Eval()
        {
            IsTraining = false;
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            throw new NotImplementedException();
        }

        public void ForwardInference(int batchSize, ReadOnlySpan<float> x, ReadOnlySpan<float> hPrev, ReadOnlySpan<float> cPrev, Span<float> hNext, Span<float> cNext)
        {
            var hSize = HiddenSize;
            var inSize = InputSize;

            using var bGatesBuf = new PooledBuffer<float>(4 * hSize, false);
            var bGates = bGatesBuf.Span;

            var wS = W.DataView.AsReadOnlySpan();
            var uS = U.DataView.AsReadOnlySpan();
            var bS = B.DataView.AsReadOnlySpan();

            for (var b = 0; b < batchSize; b++)
            {
                var bX = x.Slice(b * inSize, inSize);
                var bHPrev = hPrev.Slice(b * hSize, hSize);
                var bC = cNext.Slice(b * hSize, hSize);
                var bH = hNext.Slice(b * hSize, hSize);

                cPrev.Slice(b * hSize, hSize).CopyTo(bC);
                bS.CopyTo(bGates);

                for (var j = 0; j < 4 * hSize; j++)
                {
                    var wCol = wS.Slice(j * inSize, inSize);
                    bGates[j] += TensorPrimitives.Dot(bX, wCol);

                    var uCol = uS.Slice(j * hSize, hSize);
                    bGates[j] += TensorPrimitives.Dot(bHPrev, uCol);
                }

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

        // --- BRAKUJĄCA METODA Z IMODULE ---
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            throw new NotSupportedException("LstmCell wymaga przekazania stanów ukrytych (hidden) oraz komórki (cell). Użyj specjalnego przeciążenia Forward.");
        }

        public (AutogradNode h, AutogradNode c) Forward(ComputationGraph graph, AutogradNode x, AutogradNode h, AutogradNode c)
        {
            return TensorMath.FusedLSTMStep(graph, x, h, c, W, U, B);
        }

        public (AutogradNode h0, AutogradNode c0) ZeroState(int batchSize)
        {
            var h0 = new AutogradNode(new TensorStorage<float>(batchSize * HiddenSize), new TensorShape(batchSize, HiddenSize), false);
            var c0 = new AutogradNode(new TensorStorage<float>(batchSize * HiddenSize), new TensorShape(batchSize, HiddenSize), false);

            return (h0, c0);
        }

        private static void InitUniform(Span<float> span, float limit)
        {
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = (Random.Shared.NextSingle() * 2f - 1f) * limit;
            }
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return W;
            yield return U;
            yield return B;
        }

        public void InvalidateParameterCaches() { }
        public void Save(BinaryWriter bw) { }
        public void Load(BinaryReader br) { }
        public void Dispose()
        {
            W?.Dispose();
            U?.Dispose();
            B?.Dispose();
        }
    }
}