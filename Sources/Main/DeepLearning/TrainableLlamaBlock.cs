// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// A single <b>trainable</b> Llama/Qwen decoder block, assembled as an autograd graph forward.
    /// This is the integration of the GGUF→training bridge ops:
    /// <see cref="TensorMath.RmsNorm"/>, <see cref="TensorMath.Rope"/>,
    /// <see cref="TensorMath.ExpandKvHeads"/> (GQA), <see cref="TensorMath.Transpose01"/>,
    /// <c>TensorMath.ScaledDotProductAttention</c>, <see cref="TensorMath.SiLU"/> (SwiGLU gate)
    /// and <see cref="ComputationGraph.FrozenQuantizedLinear"/> for every projection.
    ///
    /// Pre-LN structure (bit-for-bit the same graph as the inference <c>CachedTransformerBlock</c>
    /// RMSNorm/RoPE/GQA/SwiGLU path):
    /// <code>
    ///   h  = input + Attention(RMSNorm(input, γ1))
    ///   y  = h     + SwiGLU(RMSNorm(h, γ2))
    ///   Attention: Q,K,V = frozenQuant(·);  RoPE(Q,K);  GQA-SDPA;  O = frozenQuant(·)
    ///   SwiGLU:    down( SiLU(gate(·)) ⊙ up(·) ),  gate/up/down = frozenQuant(·)
    /// </code>
    ///
    /// <b>QLoRA semantics.</b> Every projection weight is a <see cref="IDequantRowSource"/> — a frozen
    /// quantized (Q4_K / Q8) tensor straight from a GGUF, dequantized on the fly and never receiving a
    /// gradient. The only trainable parameters here are the two RMSNorm gains (γ1, γ2); LoRA adapters
    /// layer on at the projection level exactly as in the GPT-1 QLoRA path (a follow-on). Weights are
    /// combined tensors (Wq <c>[nQ·dHead, dModel]</c>, Wk/Wv <c>[nKV·dHead, dModel]</c>,
    /// Wo <c>[dModel, dModel]</c>, gate/up <c>[dFF, dModel]</c>, down <c>[dModel, dFF]</c>) — the exact
    /// layout a GGUF stores, so Session 5's loader can feed them with zero repacking.
    ///
    /// Scope: batch = 1 (T tokens in one sequence), causal mask, RoPE cos/sin supplied by the caller.
    /// </summary>
    public sealed class TrainableLlamaBlock
    {
        private readonly int _dModel;
        private readonly int _nQHeads;
        private readonly int _nKVHeads;
        private readonly int _dHead;
        private readonly int _groupSize;
        private readonly float _eps;
        private readonly bool _ropeSplitHalf;

        private readonly IDequantRowSource _wq;
        private readonly IDequantRowSource _wk;
        private readonly IDequantRowSource _wv;
        private readonly IDequantRowSource _wo;
        private readonly IDequantRowSource _wGate;
        private readonly IDequantRowSource _wUp;
        private readonly IDequantRowSource _wDown;

        /// <summary>
        /// Builds a trainable Llama block over frozen quantized projection weights.
        /// </summary>
        /// <param name="dModel">Model / residual width.</param>
        /// <param name="nQHeads">Number of query heads.</param>
        /// <param name="nKVHeads">Number of key/value heads (GQA: ≤ <paramref name="nQHeads"/>, divides it).</param>
        /// <param name="wq">Frozen Q projection, <c>[nQHeads·dHead, dModel]</c>.</param>
        /// <param name="wk">Frozen K projection, <c>[nKVHeads·dHead, dModel]</c>.</param>
        /// <param name="wv">Frozen V projection, <c>[nKVHeads·dHead, dModel]</c>.</param>
        /// <param name="wo">Frozen output projection, <c>[dModel, nQHeads·dHead]</c>.</param>
        /// <param name="wGate">Frozen SwiGLU gate, <c>[dFF, dModel]</c>.</param>
        /// <param name="wUp">Frozen SwiGLU up, <c>[dFF, dModel]</c>.</param>
        /// <param name="wDown">Frozen SwiGLU down, <c>[dModel, dFF]</c>.</param>
        /// <param name="eps">RMSNorm epsilon (Llama/Qwen: 1e-6).</param>
        /// <param name="ropeSplitHalf">RoPE pairing: <c>true</c> = split-half / HF rotate_half
        /// (Qwen2/2.5/3 GGUFs), <c>false</c> = adjacent-pair (Llama).</param>
        public TrainableLlamaBlock(
            int dModel, int nQHeads, int nKVHeads,
            IDequantRowSource wq, IDequantRowSource wk, IDequantRowSource wv, IDequantRowSource wo,
            IDequantRowSource wGate, IDequantRowSource wUp, IDequantRowSource wDown,
            float eps = 1e-6f,
            bool ropeSplitHalf = false)
        {
            if (nQHeads <= 0 || nKVHeads <= 0 || nQHeads % nKVHeads != 0)
            {
                throw new ArgumentException($"nQHeads ({nQHeads}) must be a positive multiple of nKVHeads ({nKVHeads}).");
            }
            if (wq.OutputSize % nQHeads != 0)
            {
                throw new ArgumentException($"Wq.OutputSize ({wq.OutputSize}) must be divisible by nQHeads ({nQHeads}).");
            }

            _dModel = dModel;
            _nQHeads = nQHeads;
            _nKVHeads = nKVHeads;
            _dHead = wq.OutputSize / nQHeads;
            _groupSize = nQHeads / nKVHeads;
            _eps = eps;
            _ropeSplitHalf = ropeSplitHalf;

            ValidateSource(nameof(wq), wq, nQHeads * _dHead, dModel);
            ValidateSource(nameof(wk), wk, nKVHeads * _dHead, dModel);
            ValidateSource(nameof(wv), wv, nKVHeads * _dHead, dModel);
            ValidateSource(nameof(wo), wo, dModel, nQHeads * _dHead);
            // gate/up: [dFF, dModel]; down: [dModel, dFF]. dFF inferred from gate.
            if (wGate.InputSize != dModel || wUp.InputSize != dModel || wUp.OutputSize != wGate.OutputSize)
            {
                throw new ArgumentException("SwiGLU gate/up must both map dModel → dFF with equal dFF.");
            }
            ValidateSource(nameof(wDown), wDown, dModel, wGate.OutputSize);

            _wq = wq; _wk = wk; _wv = wv; _wo = wo;
            _wGate = wGate; _wUp = wUp; _wDown = wDown;
        }

        public int DModel => _dModel;
        public int DHead => _dHead;
        public int QueryHeads => _nQHeads;
        public int KeyValueHeads => _nKVHeads;
        public int FeedForwardWidth => _wGate.OutputSize;

        /// <summary>
        /// Forward through the block, recording the autograd tape.
        /// </summary>
        /// <param name="graph">Recording graph.</param>
        /// <param name="input">Residual stream, <c>[T, dModel]</c> (batch = 1, T tokens).</param>
        /// <param name="cos">RoPE cosines, <c>[T, dHead/2]</c> (position t in row t).</param>
        /// <param name="sin">RoPE sines, <c>[T, dHead/2]</c>.</param>
        /// <param name="ln1Gamma">Trainable input-RMSNorm gain, <c>[dModel]</c>.</param>
        /// <param name="ln2Gamma">Trainable post-attention-RMSNorm gain, <c>[dModel]</c>.</param>
        public AutogradNode Forward(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode cos,
            AutogradNode sin,
            AutogradNode ln1Gamma,
            AutogradNode ln2Gamma)
            => Forward(graph, input, cos, sin, ln1Gamma, ln2Gamma, lora: null);

        /// <summary>Forward with optional trainable LoRA adapters added on top of each frozen projection
        /// (QLoRA). <paramref name="lora"/> null = base-only (norms-trainable).</summary>
        public AutogradNode Forward(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode cos,
            AutogradNode sin,
            AutogradNode ln1Gamma,
            AutogradNode ln2Gamma,
            LlamaBlockLoRA? lora)
        {
            var seqLen = input.Shape.Size / _dModel;

            // ── Attention sub-block ──────────────────────────────────────────
            var ln1 = TensorMath.RmsNorm(graph, input, ln1Gamma, _eps);             // [T, dModel]

            // Combined projections (GGUF layout) → token-major [T, heads·dHead], + optional LoRA.
            var q = Proj(graph, ln1, _wq, lora?.Q);                                 // [T, nQ·dHead]
            var k = Proj(graph, ln1, _wk, lora?.K);                                 // [T, nKV·dHead]
            var v = Proj(graph, ln1, _wv, lora?.V);                                 // [T, nKV·dHead]

            // RoPE rotates every head in the row with the position's angle (headsPerRow inferred).
            var qRot = TensorMath.Rope(graph, q, cos, sin, _ropeSplitHalf);
            var kRot = TensorMath.Rope(graph, k, cos, sin, _ropeSplitHalf);

            // token-major [T, H·dHead] → [T, H, dHead] → head-major [H, T, dHead].
            var qHM = TensorMath.Transpose01(graph, graph.Reshape(qRot, seqLen, _nQHeads, _dHead));
            var kHM = TensorMath.Transpose01(graph, graph.Reshape(kRot, seqLen, _nKVHeads, _dHead));
            var vHM = TensorMath.Transpose01(graph, graph.Reshape(v, seqLen, _nKVHeads, _dHead));

            // GQA: broadcast the KV heads up to nQHeads so the per-head SDPA runs unchanged.
            var kAttn = _groupSize > 1 ? TensorMath.ExpandKvHeads(graph, kHM, _nKVHeads, _groupSize) : kHM; // [nQ, T, dHead]
            var vAttn = _groupSize > 1 ? TensorMath.ExpandKvHeads(graph, vHM, _nKVHeads, _groupSize) : vHM;

            var oHM = TensorMath.ScaledDotProductAttention(graph, qHM, kAttn, vAttn, causalMask: true); // [nQ, T, dHead]

            // head-major [nQ, T, dHead] → [T, nQ, dHead] → [T, dModel], then the O projection.
            var oTM = graph.Reshape(TensorMath.Transpose01(graph, oHM), seqLen, _dModel);
            var attnOut = Proj(graph, oTM, _wo, lora?.O);                          // [T, dModel]

            var afterAttn = graph.Add(input, attnOut);                             // residual

            // ── SwiGLU feed-forward sub-block ────────────────────────────────
            var ln2 = TensorMath.RmsNorm(graph, afterAttn, ln2Gamma, _eps);        // [T, dModel]
            var gate = Proj(graph, ln2, _wGate, lora?.Gate);                       // [T, dFF]
            var up = Proj(graph, ln2, _wUp, lora?.Up);                             // [T, dFF]
            var gated = graph.Multiply(TensorMath.SiLU(graph, gate), up);          // SiLU(gate) ⊙ up
            var down = Proj(graph, gated, _wDown, lora?.Down);                     // [T, dModel]

            return graph.Add(afterAttn, down);                                     // residual
        }

        /// <summary>A projection through the frozen quantized base, plus an optional LoRA residual.</summary>
        private static AutogradNode Proj(ComputationGraph graph, AutogradNode x, IDequantRowSource w, LoRAAdapter? lora)
        {
            var baseOut = graph.FrozenQuantizedLinear(x, w);
            return lora is null ? baseOut : graph.Add(baseOut, lora.Apply(graph, x));
        }

        private static void ValidateSource(string name, IDequantRowSource src, int expectedOut, int expectedIn)
        {
            if (src.OutputSize != expectedOut || src.InputSize != expectedIn)
            {
                throw new ArgumentException(
                    $"{name}: expected [{expectedOut}, {expectedIn}], got [{src.OutputSize}, {src.InputSize}].");
            }
        }
    }
}
