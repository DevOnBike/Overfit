// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    ///     Single LSTM cell — processes one timestep.
    ///     Includes a zero-allocation inference path with a Fused Kernel.
    /// </summary>
    public sealed class LSTMCell : IModule
    {
        public int InputSize { get; }
        public int HiddenSize { get; }

        public AutogradNode W { get; }
        public AutogradNode U { get; }
        public AutogradNode B { get; }

        public bool IsTraining { get; private set; } = true;

        public LSTMCell(int inputSize, int hiddenSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenSize);

            InputSize = inputSize;
            HiddenSize = hiddenSize;

            var limit = MathF.Sqrt(6f / (inputSize + hiddenSize));

            W = new AutogradNode(new FastTensor<float>(inputSize, 4 * hiddenSize), true);
            U = new AutogradNode(new FastTensor<float>(hiddenSize, 4 * hiddenSize), true);
            B = new AutogradNode(new FastTensor<float>(4 * hiddenSize), true);

            InitUniform(W.Data.AsSpan(), limit);
            InitUniform(U.Data.AsSpan(), limit);

            var bSpan = B.Data.AsSpan();
            bSpan.Fill(0f);
            bSpan.Slice(0, hiddenSize).Fill(1f); // Forget gate bias
        }

        public void Train()
        {
            IsTraining = true;
        }

        public void Eval()
        {
            IsTraining = false;
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            throw new InvalidOperationException("Use Forward(graph, x, h, c) instead.");
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return W;
            yield return U;
            yield return B;
        }

        public void Save(BinaryWriter bw)
        {
            foreach (var v in W.Data.AsSpan())
            {
                bw.Write(v);
            }

            foreach (var v in U.Data.AsSpan())
            {
                bw.Write(v);
            }

            foreach (var v in B.Data.AsSpan())
            {
                bw.Write(v);
            }
        }

        public void Load(BinaryReader br)
        {
            var wSpan = W.Data.AsSpan();

            for (var i = 0; i < wSpan.Length; i++)
            {
                wSpan[i] = br.ReadSingle();
            }

            var uSpan = U.Data.AsSpan();

            for (var i = 0; i < uSpan.Length; i++)
            {
                uSpan[i] = br.ReadSingle();
            }

            var bSpan = B.Data.AsSpan();

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

        // ---------------------------------------------------------------------------
        // Forward — Zero-Allocation Inference Path (Production Fast-Path)
        // ---------------------------------------------------------------------------

        public void ForwardInference(int batchSize, ReadOnlySpan<float> x, Span<float> h, Span<float> c)
        {
            var hSize = HiddenSize;
            var gatesLen = batchSize * 4 * hSize;

            var gatesArr = ArrayPool<float>.Shared.Rent(gatesLen);
            var uhArr = ArrayPool<float>.Shared.Rent(gatesLen);

            try
            {
                var gates = gatesArr.AsSpan(0, gatesLen);
                var uh = uhArr.AsSpan(0, gatesLen);

                var wSpan = W.Data.AsReadOnlySpan();
                var uSpan = U.Data.AsReadOnlySpan();
                var bSpan = B.Data.AsReadOnlySpan();

                gates.Clear();
                uh.Clear();

                for (var b = 0; b < batchSize; b++)
                {
                    var bGates = gates.Slice(b * 4 * hSize, 4 * hSize);
                    var bX = x.Slice(b * InputSize, InputSize);

                    // 1. wx = x * W
                    for (var i = 0; i < InputSize; i++)
                    {
                        var xVal = bX[i];

                        if (xVal != 0f)
                        {
                            TensorPrimitives.MultiplyAdd(wSpan.Slice(i * 4 * hSize, 4 * hSize), xVal, bGates, bGates);
                        }
                    }

                    // 2. wx = wx + B
                    TensorPrimitives.Add(bGates, bSpan, bGates);

                    var bUh = uh.Slice(b * 4 * hSize, 4 * hSize);
                    var bH = h.Slice(b * hSize, hSize);

                    // 3. uh = h * U
                    for (var i = 0; i < hSize; i++)
                    {
                        var hVal = bH[i];

                        if (hVal != 0f)
                        {
                            TensorPrimitives.MultiplyAdd(uSpan.Slice(i * 4 * hSize, 4 * hSize), hVal, bUh, bUh);
                        }
                    }

                    // 4. gates = wx + uh
                    TensorPrimitives.Add(bGates, bUh, bGates);

                    // 5. Fused Activation and State Update (100% SIMD)
                    var bC = c.Slice(b * hSize, hSize);

                    var gF = bGates.Slice(0, hSize);
                    var gI = bGates.Slice(hSize, hSize);
                    var gG = bGates.Slice(2 * hSize, hSize);
                    var gO = bGates.Slice(3 * hSize, hSize);

                    TensorPrimitives.Sigmoid(gF, gF);
                    TensorPrimitives.Sigmoid(gI, gI);
                    TensorPrimitives.Tanh(gG, gG);
                    TensorPrimitives.Sigmoid(gO, gO);

                    // C = f * C + i * g
                    TensorPrimitives.Multiply(gF, bC, bC);
                    TensorPrimitives.MultiplyAdd(gI, gG, bC, bC);

                    // H = o * tanh(C)
                    TensorPrimitives.Tanh(bC, gG); // Reuse gG as a free temp buffer
                    TensorPrimitives.Multiply(gO, gG, bH);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(gatesArr);
                ArrayPool<float>.Shared.Return(uhArr);
            }
        }

        public (AutogradNode h, AutogradNode c) Forward(ComputationGraph graph, AutogradNode x, AutogradNode h, AutogradNode c)
        {
            var wx = TensorMath.Linear(graph, x, W, B);
            var uh = TensorMath.MatMul(graph, h, U);
            var gates = TensorMath.Add(graph, wx, uh);

            var gF = TensorMath.GateSlice(graph, gates, HiddenSize, 0);
            var gI = TensorMath.GateSlice(graph, gates, HiddenSize, 1);
            var gG = TensorMath.GateSlice(graph, gates, HiddenSize, 2);
            var gO = TensorMath.GateSlice(graph, gates, HiddenSize, 3);

            var fT = TensorMath.Sigmoid(graph, gF);
            var iT = TensorMath.Sigmoid(graph, gI);
            var gT = TensorMath.Tanh(graph, gG);
            var oT = TensorMath.Sigmoid(graph, gO);

            var fc = TensorMath.Multiply(graph, fT, c);
            var ig = TensorMath.Multiply(graph, iT, gT);
            var cNew = TensorMath.Add(graph, fc, ig);

            var hNew = TensorMath.Multiply(graph, oT, TensorMath.Tanh(graph, cNew));

            return (hNew, cNew);
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
    }
}