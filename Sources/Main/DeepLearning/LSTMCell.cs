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
        public int InputSize
        {
            get;
        }
        public int HiddenSize
        {
            get;
        }
        public AutogradNode W
        {
            get;
        }
        public AutogradNode U
        {
            get;
        }
        public AutogradNode B
        {
            get;
        }
        public bool IsTraining { get; private set; } = true;

        public LstmCell(int inputSize, int hiddenSize)
        {
            InputSize = inputSize;
            HiddenSize = hiddenSize;
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

        // --- MISSING METHOD FROM IMODULE ---
        public AutogradNode Forward(ComputationGraph? graph, AutogradNode input)
        {
            throw new OverfitRuntimeException("LstmCell requires the hidden state and cell state to be passed. Use the dedicated Forward overload.");
        }

        public (AutogradNode h, AutogradNode c) Forward(ComputationGraph? graph, AutogradNode x, AutogradNode h, AutogradNode c)
        {
            return TensorMath.FusedLSTMStep(graph, x, h, c, W, U, B);
        }

        public (AutogradNode h0, AutogradNode c0) ZeroState(int batchSize)
        {
            var h0 = new AutogradNode(new TensorStorage<float>(batchSize * HiddenSize), new TensorShape(batchSize, HiddenSize));
            var c0 = new AutogradNode(new TensorStorage<float>(batchSize * HiddenSize), new TensorShape(batchSize, HiddenSize));

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

        public void InvalidateParameterCaches()
        {
        }
        /// <summary>
        /// Writes the three parameter tensors, in <see cref="Parameters"/> order.
        ///
        /// <para><b>These two methods were empty bodies</b> - not incomplete, <c>{ }</c> - until 2026-08-02,
        /// and <c>LstmLayer</c>, <c>LstmAutoencoder</c> and <c>Crnn</c> all delegate through them. The result
        /// was a shipped capability broken in the way that is hardest to notice: save a trained CRNN, load it
        /// back, and the convolutions, norms and classifier all return correctly while the recurrent core
        /// sits at its random initialisation. Nothing throws, the model reports as loaded, and it produces
        /// nonsense. The OCR demo that reads digits at loss 0.006 could not survive a round-trip through
        /// disk.</para>
        ///
        /// <para>Order matters and must match <see cref="Load"/>: a composite <c>Save</c> writes one stream,
        /// so a mismatch here misaligns every layer that follows rather than failing here.</para>
        /// </summary>
        public void Save(BinaryWriter bw)
        {
            ArgumentNullException.ThrowIfNull(bw);

            WriteTensor(bw, W);
            WriteTensor(bw, U);
            WriteTensor(bw, B);
        }

        /// <inheritdoc cref="Save"/>
        public void Load(BinaryReader br)
        {
            ArgumentNullException.ThrowIfNull(br);

            ReadTensor(br, W);
            ReadTensor(br, U);
            ReadTensor(br, B);
        }

        /// <summary>Length then floats — the same wire format <c>Parameter.Save</c> writes, so a checkpoint
        /// written by either side is readable by the other.</summary>
        private static void WriteTensor(BinaryWriter bw, AutogradNode node)
        {
            var data = node.DataView.AsReadOnlySpan();

            bw.Write(data.Length);

            for (var i = 0; i < data.Length; i++)
            {
                bw.Write(data[i]);
            }
        }

        /// <inheritdoc cref="WriteTensor"/>
        private static void ReadTensor(BinaryReader br, AutogradNode node)
        {
            var data = node.DataView.AsSpan();
            var length = br.ReadInt32();

            if (length != data.Length)
            {
                throw new OverfitFormatException(
                    $"Checkpoint size {length} does not match LSTM parameter size {data.Length}.");
            }

            for (var i = 0; i < length; i++)
            {
                data[i] = br.ReadSingle();
            }
        }
        public void Dispose()
        {
            W?.Dispose();
            U?.Dispose();
            B?.Dispose();
        }
    }
}