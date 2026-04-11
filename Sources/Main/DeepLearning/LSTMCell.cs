// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Single LSTM cell — processes one timestep.
    ///
    /// Gates (all share the same concatenated input [h_{t-1}, x_t]):
    ///   f_t = σ(W_f · [h, x] + b_f)          forget gate  — what to erase from cell state
    ///   i_t = σ(W_i · [h, x] + b_i)          input gate   — what to write to cell state
    ///   g_t = tanh(W_g · [h, x] + b_g)       candidate    — what to potentially write
    ///   o_t = σ(W_o · [h, x] + b_o)          output gate  — what to expose as hidden state
    ///
    ///   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t     cell state update
    ///   h_t = o_t ⊙ tanh(c_t)                hidden state update
    ///
    /// Weights are packed into two matrices for efficient matmul:
    ///   W_gates [4 * hiddenSize, inputSize]   — input projection
    ///   U_gates [4 * hiddenSize, hiddenSize]  — hidden state projection
    ///   b_gates [4 * hiddenSize]              — biases
    ///
    /// Gate order in packed matrices: f, i, g, o (index * hiddenSize offsets).
    /// </summary>
    public sealed class LSTMCell : IModule
    {
        private readonly int _inputSize;
        private readonly int _hiddenSize;

        // W [4*hiddenSize, inputSize]  — projects input x_t
        public AutogradNode W { get; }

        // U [4*hiddenSize, hiddenSize] — projects previous hidden state h_{t-1}
        public AutogradNode U { get; }

        // b [4*hiddenSize]             — biases
        public AutogradNode B { get; }

        public bool IsTraining { get; private set; } = true;

        public LSTMCell(int inputSize, int hiddenSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenSize);

            _inputSize = inputSize;
            _hiddenSize = hiddenSize;

            // Xavier uniform initialization — good for sigmoid/tanh gates
            var limit = MathF.Sqrt(6f / (inputSize + hiddenSize));

            W = new AutogradNode(new FastTensor<float>(4 * hiddenSize, inputSize), requiresGrad: true);
            U = new AutogradNode(new FastTensor<float>(4 * hiddenSize, hiddenSize), requiresGrad: true);
            B = new AutogradNode(new FastTensor<float>(4 * hiddenSize), requiresGrad: true);

            InitUniform(W.Data.AsSpan(), limit);
            InitUniform(U.Data.AsSpan(), limit);

            // Forget gate bias initialised to 1 — helps gradient flow early in training
            var bSpan = B.Data.AsSpan();
            bSpan.Fill(0f);
            bSpan.Slice(0, hiddenSize).Fill(1f);
        }

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        // ---------------------------------------------------------------------------
        // Forward — processes one timestep
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Processes a single timestep.
        /// </summary>
        /// <param name="graph">Computation graph — null for inference.</param>
        /// <param name="x">Input at current timestep [batch, inputSize].</param>
        /// <param name="h">Previous hidden state [batch, hiddenSize].</param>
        /// <param name="c">Previous cell state [batch, hiddenSize].</param>
        /// <returns>Updated (h_t, c_t).</returns>
        public (AutogradNode h, AutogradNode c) Forward(
            ComputationGraph graph,
            AutogradNode x,
            AutogradNode h,
            AutogradNode c)
        {
            // gates = W·x + U·h + b  →  [batch, 4*hiddenSize]
            var wx = TensorMath.Linear(graph, x, W, B);
            var uh = TensorMath.MatMul(graph, h, U);
            var gates = TensorMath.Add(graph, wx, uh);

            // Split gates along feature axis into 4 slices of [batch, hiddenSize]
            var gF = TensorMath.GateSlice(graph, gates, _hiddenSize, 0);   // forget
            var gI = TensorMath.GateSlice(graph, gates, _hiddenSize, 1);   // input
            var gG = TensorMath.GateSlice(graph, gates, _hiddenSize, 2);   // candidate
            var gO = TensorMath.GateSlice(graph, gates, _hiddenSize, 3);   // output

            var fT = TensorMath.Sigmoid(graph, gF);
            var iT = TensorMath.Sigmoid(graph, gI);
            var gT = TensorMath.Tanh(graph, gG);
            var oT = TensorMath.Sigmoid(graph, gO);

            // c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
            var fc = TensorMath.Multiply(graph, fT, c);
            var ig = TensorMath.Multiply(graph, iT, gT);
            var cNew = TensorMath.Add(graph, fc, ig);

            // h_t = o_t ⊙ tanh(c_t)
            var hNew = TensorMath.Multiply(graph, oT, TensorMath.Tanh(graph, cNew));

            return (hNew, cNew);
        }

        /// <summary>
        /// Creates zero-initialised hidden and cell states for the start of a sequence.
        /// </summary>
        public (AutogradNode h0, AutogradNode c0) ZeroState(int batchSize)
        {
            var h0 = new AutogradNode(new FastTensor<float>(batchSize, _hiddenSize), requiresGrad: false);
            var c0 = new AutogradNode(new FastTensor<float>(batchSize, _hiddenSize), requiresGrad: false);
            return (h0, c0);
        }

        // ---------------------------------------------------------------------------
        // IModule
        // ---------------------------------------------------------------------------

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
            => throw new InvalidOperationException(
                "LSTMCell requires explicit h and c states. Use Forward(graph, x, h, c) instead.");

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return W;
            yield return U;
            yield return B;
        }

        public void Save(BinaryWriter bw)
        {
            foreach (var v in W.Data.AsSpan()) bw.Write(v);
            foreach (var v in U.Data.AsSpan()) bw.Write(v);
            foreach (var v in B.Data.AsSpan()) bw.Write(v);
        }

        public void Load(BinaryReader br)
        {
            var wSpan = W.Data.AsSpan();
            for (var i = 0; i < wSpan.Length; i++) wSpan[i] = br.ReadSingle();

            var uSpan = U.Data.AsSpan();
            for (var i = 0; i < uSpan.Length; i++) uSpan[i] = br.ReadSingle();

            var bSpan = B.Data.AsSpan();
            for (var i = 0; i < bSpan.Length; i++) bSpan[i] = br.ReadSingle();
        }

        public void Dispose()
        {
            W?.Dispose();
            U?.Dispose();
            B?.Dispose();
        }

        // ---------------------------------------------------------------------------

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