// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.LanguageModels.Rope;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;

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

        public int FeedForwardWidthCached => _wGate.OutputSize;
        public int KvWidth => _nKVHeads * _dHead;

        /// <summary>
        /// Incremental cached decode of ONE token (no autograd, no gradients) — the fast-generation path.
        /// Processes only the new token through the block, attending its query over the KV cache built from
        /// all earlier positions (so generation is O(1) layer-work per token instead of re-running the whole
        /// sequence). The new token's K/V are appended to <paramref name="cacheK"/>/<paramref name="cacheV"/>
        /// at <paramref name="position"/>. Same math as the training <c>Forward</c> (RMSNorm → QKV+LoRA → RoPE →
        /// GQA attention → O+LoRA → residual → RMSNorm → SwiGLU+LoRA → residual), so it matches the trained
        /// model's <c>Forward</c>. Naive per-row dequant kernels (not the optimized inference GEMV — see
        /// ROADMAP "Fast fine-tuned decode").
        /// </summary>
        public void DecodeStep(
            ReadOnlySpan<float> input,   // [dModel] hidden for the new token
            int position,
            RopeTable rope,
            AutogradNode ln1Gamma,
            AutogradNode ln2Gamma,
            LlamaBlockLoRA? lora,
            Span<float> cacheK,          // [maxLen * KvWidth] for this layer
            Span<float> cacheV,
            Span<float> output)          // [dModel]
        {
            var dModel = _dModel;
            var dHead = _dHead;
            var qDim = _nQHeads * dHead;
            var kvDim = _nKVHeads * dHead;
            var dFF = _wGate.OutputSize;
            var scale = 1f / MathF.Sqrt(dHead);
            var len = position + 1;

            using var ln1B = new PooledBuffer<float>(dModel, clearMemory: false);
            using var qB = new PooledBuffer<float>(qDim, false);
            using var kB = new PooledBuffer<float>(kvDim, false);
            using var vB = new PooledBuffer<float>(kvDim, false);
            using var attnB = new PooledBuffer<float>(qDim, false);
            using var oB = new PooledBuffer<float>(dModel, false);
            using var afterB = new PooledBuffer<float>(dModel, false);
            using var ln2B = new PooledBuffer<float>(dModel, false);
            using var gateB = new PooledBuffer<float>(dFF, false);
            using var upB = new PooledBuffer<float>(dFF, false);
            using var downB = new PooledBuffer<float>(dModel, false);
            using var rowB = new PooledBuffer<float>(Math.Max(dModel, dFF), false);
            using var scoresB = new PooledBuffer<float>(len, false);

            Span<float> ln1 = ln1B.Span, q = qB.Span, k = kB.Span, v = vB.Span, attn = attnB.Span;
            Span<float> o = oB.Span, afterAttn = afterB.Span, ln2 = ln2B.Span;
            Span<float> gate = gateB.Span, up = upB.Span, down = downB.Span, scores = scoresB.Span;
            var row = rowB.Span;

            // ── attention ──
            RmsNormVec(input, ln1Gamma.DataView.AsReadOnlySpan(), ln1, _eps);
            ProjVec(ln1, _wq, lora?.Q, q, row);
            ProjVec(ln1, _wk, lora?.K, k, row);
            ProjVec(ln1, _wv, lora?.V, v, row);

            for (var h = 0; h < _nQHeads; h++) { RopeKernel.Apply(q.Slice(h * dHead, dHead), rope, position); }
            for (var h = 0; h < _nKVHeads; h++) { RopeKernel.Apply(k.Slice(h * dHead, dHead), rope, position); }

            k.CopyTo(cacheK.Slice(position * kvDim, kvDim));
            v.CopyTo(cacheV.Slice(position * kvDim, kvDim));

            for (var h = 0; h < _nQHeads; h++)
            {
                var kvh = h / _groupSize;
                var qh = q.Slice(h * dHead, dHead);
                for (var j = 0; j < len; j++)
                {
                    scores[j] = TensorPrimitives.Dot(qh, cacheK.Slice(j * kvDim + kvh * dHead, dHead)) * scale;
                }
                Softmax(scores.Slice(0, len));
                var outH = attn.Slice(h * dHead, dHead);
                outH.Clear();
                for (var j = 0; j < len; j++)
                {
                    TensorPrimitives.MultiplyAdd(cacheV.Slice(j * kvDim + kvh * dHead, dHead), scores[j], outH, outH);
                }
            }

            ProjVec(attn, _wo, lora?.O, o, row);
            TensorPrimitives.Add(input, o, afterAttn);

            // ── SwiGLU FFN ──
            RmsNormVec(afterAttn, ln2Gamma.DataView.AsReadOnlySpan(), ln2, _eps);
            ProjVec(ln2, _wGate, lora?.Gate, gate, row);
            ProjVec(ln2, _wUp, lora?.Up, up, row);
            for (var i = 0; i < dFF; i++) { gate[i] = gate[i] / (1f + MathF.Exp(-gate[i])) * up[i]; }
            ProjVec(gate, _wDown, lora?.Down, down, row);
            TensorPrimitives.Add(afterAttn, down, output);
        }

        // ── plain-span inference helpers (cached decode only) ──

        private static void RmsNormVec(ReadOnlySpan<float> x, ReadOnlySpan<float> gamma, Span<float> dst, float eps)
        {
            var inv = 1f / MathF.Sqrt(TensorPrimitives.Dot(x, x) / x.Length + eps);
            for (var i = 0; i < x.Length; i++) { dst[i] = x[i] * inv * gamma[i]; }
        }

        /// <summary>out[o] = Σ_i dequant(W)[o,i]·x[i] (+ optional (x·A)·B). <paramref name="row"/> is dequant scratch.</summary>
        private static void ProjVec(ReadOnlySpan<float> x, IDequantRowSource w, LoRAAdapter? lora, Span<float> dst, Span<float> row)
        {
            var outDim = w.OutputSize;
            var inDim = w.InputSize;
            var rowS = row.Slice(0, inDim);
            for (var oIdx = 0; oIdx < outDim; oIdx++)
            {
                w.DecodeRow(oIdx, rowS);
                dst[oIdx] = TensorPrimitives.Dot(rowS, x);
            }

            if (lora is null) { return; }
            var rank = lora.Rank;
            var a = lora.A.DataView.AsReadOnlySpan(); // [inDim, rank]
            var b = lora.B.DataView.AsReadOnlySpan(); // [rank, outDim]
            Span<float> tmp = stackalloc float[rank];
            tmp.Clear();
            for (var i = 0; i < inDim; i++)
            {
                var xi = x[i];
                if (xi == 0f) { continue; }
                var aRow = a.Slice(i * rank, rank);
                for (var r = 0; r < rank; r++) { tmp[r] += xi * aRow[r]; }
            }
            for (var r = 0; r < rank; r++)
            {
                var tr = tmp[r];
                if (tr == 0f) { continue; }
                TensorPrimitives.MultiplyAdd(b.Slice(r * outDim, outDim), tr, dst, dst);
            }
        }

        private static void Softmax(Span<float> s)
        {
            var max = float.NegativeInfinity;
            for (var i = 0; i < s.Length; i++) { if (s[i] > max) { max = s[i]; } }
            var sum = 0f;
            for (var i = 0; i < s.Length; i++) { s[i] = MathF.Exp(s[i] - max); sum += s[i]; }
            var inv = 1f / sum;
            for (var i = 0; i < s.Length; i++) { s[i] *= inv; }
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
