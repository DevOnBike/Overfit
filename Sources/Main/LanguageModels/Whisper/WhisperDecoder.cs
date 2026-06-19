// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Maths;

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
    /// <summary>
    /// Whisper text decoder (inference, plain-span): token-embedding + positional-embedding → N pre-LN blocks,
    /// each = <b>causal</b> self-attention + <b>cross-attention</b> to the encoder output + GELU MLP → final
    /// LayerNorm → logits over the vocab (LM head is the tied token embedding). <see cref="DecodeCached"/>
    /// is the fast path (KV cache: cross-attn K/V projected once, self-attn K/V cached incrementally,
    /// allocation-free per-token step); <see cref="Decode"/>/<see cref="Forward"/> are the simple reference
    /// path that recomputes the whole sequence each step.
    /// </summary>
    public sealed class WhisperDecoder
    {
        // All weights pre-resolved once (no per-step string lookups → the cached decode step is zero-alloc).
        private sealed class Layer
        {
            public required float[] AttnLnW, AttnLnB, AttnQW, AttnQB, AttnKW, AttnVW, AttnVB, AttnOW, AttnOB;
            public required float[] CrossLnW, CrossLnB, CrossQW, CrossQB, CrossKW, CrossVW, CrossVB, CrossOW, CrossOB;
            public required float[] MlpLnW, MlpLnB, Mlp0W, Mlp0B, Mlp2W, Mlp2B;
        }

        private readonly WhisperModel _m;
        private readonly int _nState;
        private readonly int _nHead;
        private readonly int _nLayer;
        private readonly int _nVocab;
        private readonly int _dFF;
        private readonly float[] _tokenEmbedding; // [nVocab, nState]
        private readonly float[] _posEmb;
        private readonly float[] _lnW, _lnB;
        private readonly Layer[] _layers;

        public WhisperDecoder(WhisperModel model)
        {
            _m = model;
            _nState = model.Config.NTextState;
            _nHead = model.Config.NTextHead;
            _nLayer = model.Config.NTextLayer;
            _nVocab = model.Config.NVocab;
            _tokenEmbedding = T("decoder.token_embedding.weight");
            _posEmb = T("decoder.positional_embedding");
            _lnW = T("decoder.ln.weight");
            _lnB = T("decoder.ln.bias");
            _dFF = _m.Tensors["decoder.blocks.0.mlp.0.weight"].Shape[0];

            _layers = new Layer[_nLayer];
            for (var b = 0; b < _nLayer; b++)
            {
                var p = $"decoder.blocks.{b}.";
                _layers[b] = new Layer
                {
                    AttnLnW = T(p + "attn_ln.weight"),
                    AttnLnB = T(p + "attn_ln.bias"),
                    AttnQW = T(p + "attn.query.weight"),
                    AttnQB = T(p + "attn.query.bias"),
                    AttnKW = T(p + "attn.key.weight"),
                    AttnVW = T(p + "attn.value.weight"),
                    AttnVB = T(p + "attn.value.bias"),
                    AttnOW = T(p + "attn.out.weight"),
                    AttnOB = T(p + "attn.out.bias"),
                    CrossLnW = T(p + "cross_attn_ln.weight"),
                    CrossLnB = T(p + "cross_attn_ln.bias"),
                    CrossQW = T(p + "cross_attn.query.weight"),
                    CrossQB = T(p + "cross_attn.query.bias"),
                    CrossKW = T(p + "cross_attn.key.weight"),
                    CrossVW = T(p + "cross_attn.value.weight"),
                    CrossVB = T(p + "cross_attn.value.bias"),
                    CrossOW = T(p + "cross_attn.out.weight"),
                    CrossOB = T(p + "cross_attn.out.bias"),
                    MlpLnW = T(p + "mlp_ln.weight"),
                    MlpLnB = T(p + "mlp_ln.bias"),
                    Mlp0W = T(p + "mlp.0.weight"),
                    Mlp0B = T(p + "mlp.0.bias"),
                    Mlp2W = T(p + "mlp.2.weight"),
                    Mlp2B = T(p + "mlp.2.bias"),
                };
            }
        }

        /// <summary>
        /// Forward over <paramref name="tokens"/> attending the fixed encoder output
        /// (<paramref name="encoderOut"/> <c>[nCtx × nState]</c>); returns the next-token logits for the LAST
        /// position (<c>[nVocab]</c>).
        /// </summary>
#pragma warning disable OVERFIT001 // Reference decode path (recompute-all, per-step buffers): the serving path is DecodeCached/Step, which is allocation-free. Kept simple for correctness/parity.
        public float[] Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<float> encoderOut, int nCtx)
        {
            var seq = tokens.Length;
            var pos = T("decoder.positional_embedding");

            // token embedding + positional embedding
            var x = new float[seq * _nState];
            for (var t = 0; t < seq; t++)
            {
                var emb = _tokenEmbedding.AsSpan(tokens[t] * _nState, _nState);
                var p = pos.AsSpan(t * _nState, _nState);
                for (var c = 0; c < _nState; c++) { x[t * _nState + c] = emb[c] + p[c]; }
            }

            var ln = new float[seq * _nState];
            var tmp = new float[seq * _nState];
            for (var b = 0; b < _nLayer; b++)
            {
                var p = $"decoder.blocks.{b}.";

                // causal self-attention
                WhisperKernels.LayerNorm(x, T(p + "attn_ln.weight"), T(p + "attn_ln.bias"), ln, seq, _nState);
                WhisperKernels.MultiHeadAttention(
                    ln, seq, ln, seq, _nState, _nHead,
                    T(p + "attn.query.weight"), T(p + "attn.query.bias"),
                    T(p + "attn.key.weight"),
                    T(p + "attn.value.weight"), T(p + "attn.value.bias"),
                    T(p + "attn.out.weight"), T(p + "attn.out.bias"),
                    tmp, causal: true);
                TensorPrimitives.Add(x, tmp, x);

                // cross-attention to the encoder output (non-causal)
                WhisperKernels.LayerNorm(x, T(p + "cross_attn_ln.weight"), T(p + "cross_attn_ln.bias"), ln, seq, _nState);
                WhisperKernels.MultiHeadAttention(
                    ln, seq, encoderOut, nCtx, _nState, _nHead,
                    T(p + "cross_attn.query.weight"), T(p + "cross_attn.query.bias"),
                    T(p + "cross_attn.key.weight"),
                    T(p + "cross_attn.value.weight"), T(p + "cross_attn.value.bias"),
                    T(p + "cross_attn.out.weight"), T(p + "cross_attn.out.bias"),
                    tmp, causal: false);
                TensorPrimitives.Add(x, tmp, x);

                // MLP
                WhisperKernels.LayerNorm(x, T(p + "mlp_ln.weight"), T(p + "mlp_ln.bias"), ln, seq, _nState);
                var w0 = _m.Tensors[p + "mlp.0.weight"];
                var dFF = w0.Shape[0];
                var hidden = new float[seq * dFF];
                WhisperKernels.Linear(ln, w0.Data, T(p + "mlp.0.bias"), hidden, seq, _nState, dFF);
                WhisperKernels.GeluInPlace(hidden);
                WhisperKernels.Linear(hidden, T(p + "mlp.2.weight"), T(p + "mlp.2.bias"), tmp, seq, dFF, _nState);
                TensorPrimitives.Add(x, tmp, x);
            }

            WhisperKernels.LayerNorm(x, T("decoder.ln.weight"), T("decoder.ln.bias"), x, seq, _nState);

            // logits for the last position: x_last @ token_embeddingᵀ (tied LM head).
            var last = x.AsSpan((seq - 1) * _nState, _nState);
            var logits = new float[_nVocab];
            WhisperKernels.Linear(last, _tokenEmbedding, ReadOnlySpan<float>.Empty, logits, 1, _nState, _nVocab);
            return logits;
        }
#pragma warning restore OVERFIT001

        /// <summary>
        /// Greedy autoregressive decode: starts from <paramref name="promptTokens"/> (e.g.
        /// <c>[sot, lang, transcribe, notimestamps]</c>) and appends argmax tokens until
        /// <paramref name="endOfTranscript"/> or <paramref name="maxNewTokens"/>. Returns the generated tokens
        /// (excluding the prompt).
        /// </summary>
        public int[] Decode(
            ReadOnlySpan<float> encoderOut, int nCtx, ReadOnlySpan<int> promptTokens,
            int endOfTranscript, int maxNewTokens)
        {
            var tokens = new List<int>(promptTokens.Length + maxNewTokens);
            for (var i = 0; i < promptTokens.Length; i++) { tokens.Add(promptTokens[i]); }
            var produced = new List<int>();

            for (var i = 0; i < maxNewTokens; i++)
            {
                var logits = Forward(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(tokens), encoderOut, nCtx);
                var next = MathUtils.ArgMax(logits);
                if (next == endOfTranscript) { break; }
                tokens.Add(next);
                produced.Add(next);
            }
            return produced.ToArray();
        }

        // ── KV-cache decode: cross-attn K/V computed once; self-attn K/V cached incrementally; the
        //    per-token step is allocation-free (all scratch lives in the pre-allocated state). ──

        internal sealed class State
        {
            // Caches are single CONTIGUOUS buffers (layer-major), not jagged arrays — one allocation,
            // cache-friendly. Layer b's slice is [b*stride, stride): cross stride = NCtx*nState, self = MaxLen*nState.
            public required float[] CrossK;   // nLayer * NCtx  * nState — projected once
            public required float[] CrossV;
            public required float[] SelfK;     // nLayer * MaxLen * nState — grown per step
            public required float[] SelfV;
            public required int NCtx;
            public required int MaxLen;
            public int Position;
            // per-step scratch (reused; no per-step allocation)
            public required float[] X, Ln, Q, K, V, Attn, Proj, Hidden, Scores, Logits;
        }

        /// <summary>
        /// Greedy decode with a KV cache — O(n) per token instead of re-running the whole sequence. Same
        /// greedy result as <see cref="Decode"/> but far faster (cross-attention K/V are projected once from
        /// the fixed encoder output; self-attention K/V accumulate in a cache; the per-token step is
        /// allocation-free). Returns the generated tokens (excluding the prompt).
        /// </summary>
        public int[] DecodeCached(
            ReadOnlySpan<float> encoderOut, int nCtx, ReadOnlySpan<int> promptTokens,
            int endOfTranscript, int maxNewTokens)
        {
            var maxLen = promptTokens.Length + maxNewTokens;
            var state = CreateState(encoderOut, nCtx, maxLen);

            // Prefill the prompt (last prompt token's logits start generation).
            ReadOnlySpan<float> logits = default;
            for (var i = 0; i < promptTokens.Length; i++) { logits = Step(state, promptTokens[i]); }

            var produced = new List<int>(maxNewTokens);
            for (var i = 0; i < maxNewTokens; i++)
            {
                var next = MathUtils.ArgMax(logits);
                if (next == endOfTranscript) { break; }
                produced.Add(next);
                if (produced.Count >= maxNewTokens) { break; }
                logits = Step(state, next);
            }
            return produced.ToArray();
        }

        private State? _reuseState; // reused across DecodeCached calls (streaming) when dims match

        internal State CreateState(ReadOnlySpan<float> encoderOut, int nCtx, int maxLen)
        {
            var n = _nState;
            var crossStride = nCtx * n;
            var selfStride = maxLen * n;

            var s = _reuseState;
            if (s is null || s.NCtx != nCtx || s.MaxLen != maxLen)
            {
                var scoreLen = Math.Max(nCtx, maxLen);
#pragma warning disable OVERFIT001 // Decode-state scratch allocated once per (reused) State — _reuseState keeps it across streaming steps; the per-token Step is allocation-free.
                s = new State
                {
                    CrossK = new float[_nLayer * crossStride],
                    CrossV = new float[_nLayer * crossStride],
                    SelfK = new float[_nLayer * selfStride],
                    SelfV = new float[_nLayer * selfStride],
                    NCtx = nCtx,
                    MaxLen = maxLen,
                    X = new float[n],
                    Ln = new float[n],
                    Q = new float[n],
                    K = new float[n],
                    V = new float[n],
                    Attn = new float[n],
                    Proj = new float[n],
                    Hidden = new float[_dFF],
                    Scores = new float[scoreLen],
                    Logits = new float[_nVocab],
                };
#pragma warning restore OVERFIT001
                _reuseState = s;
            }

            s.Position = 0;
            // Cross-attention K/V are re-projected from this call's encoder output; self-attention K/V are
            // (re)written per step before being read, so the reused buffers need no clearing.
            for (var b = 0; b < _nLayer; b++)
            {
                var l = _layers[b];
                WhisperKernels.Linear(encoderOut, l.CrossKW, ReadOnlySpan<float>.Empty, s.CrossK.AsSpan(b * crossStride, crossStride), nCtx, n, n);
                WhisperKernels.Linear(encoderOut, l.CrossVW, l.CrossVB, s.CrossV.AsSpan(b * crossStride, crossStride), nCtx, n, n);
            }
            return s;
        }

        internal ReadOnlySpan<float> Step(State s, int tokenId)
        {
            var n = _nState;
            var t = s.Position;
            var x = s.X.AsSpan(0, n);

            // token + positional embedding for the single new token
            var emb = _tokenEmbedding.AsSpan(tokenId * n, n);
            var pos = _posEmb.AsSpan(t * n, n);
            for (var c = 0; c < n; c++) { x[c] = emb[c] + pos[c]; }

            var crossStride = s.NCtx * n;
            var selfStride = s.MaxLen * n;
            Span<float> ln = s.Ln, q = s.Q, k = s.K, v = s.V, attn = s.Attn, proj = s.Proj;
            for (var b = 0; b < _nLayer; b++)
            {
                var l = _layers[b];
                var selfBase = b * selfStride;
                var crossBase = b * crossStride;

                // causal self-attention over the growing cache
                WhisperKernels.LayerNorm(x, l.AttnLnW, l.AttnLnB, ln, 1, n);
                WhisperKernels.Linear(ln, l.AttnQW, l.AttnQB, q, 1, n, n);
                WhisperKernels.Linear(ln, l.AttnKW, ReadOnlySpan<float>.Empty, k, 1, n, n);
                WhisperKernels.Linear(ln, l.AttnVW, l.AttnVB, v, 1, n, n);
                k.CopyTo(s.SelfK.AsSpan(selfBase + t * n, n));
                v.CopyTo(s.SelfV.AsSpan(selfBase + t * n, n));
                WhisperKernels.SingleQueryAttention(q, s.SelfK.AsSpan(selfBase, (t + 1) * n), s.SelfV.AsSpan(selfBase, (t + 1) * n), t + 1, n, _nHead, s.Scores, attn);
                WhisperKernels.Linear(attn, l.AttnOW, l.AttnOB, proj, 1, n, n);
                TensorPrimitives.Add(x, proj, x);

                // cross-attention over the precomputed encoder K/V
                WhisperKernels.LayerNorm(x, l.CrossLnW, l.CrossLnB, ln, 1, n);
                WhisperKernels.Linear(ln, l.CrossQW, l.CrossQB, q, 1, n, n);
                WhisperKernels.SingleQueryAttention(q, s.CrossK.AsSpan(crossBase, crossStride), s.CrossV.AsSpan(crossBase, crossStride), s.NCtx, n, _nHead, s.Scores, attn);
                WhisperKernels.Linear(attn, l.CrossOW, l.CrossOB, proj, 1, n, n);
                TensorPrimitives.Add(x, proj, x);

                // MLP
                WhisperKernels.LayerNorm(x, l.MlpLnW, l.MlpLnB, ln, 1, n);
                WhisperKernels.Linear(ln, l.Mlp0W, l.Mlp0B, s.Hidden, 1, n, _dFF);
                WhisperKernels.GeluInPlace(s.Hidden);
                WhisperKernels.Linear(s.Hidden, l.Mlp2W, l.Mlp2B, proj, 1, _dFF, n);
                TensorPrimitives.Add(x, proj, x);
            }

            WhisperKernels.LayerNorm(x, _lnW, _lnB, x, 1, n);
            WhisperKernels.Linear(x, _tokenEmbedding, ReadOnlySpan<float>.Empty, s.Logits, 1, n, _nVocab);
            s.Position++;
            return s.Logits;
        }

        private float[] T(string name)
            => _m.Tensors.TryGetValue(name, out var t)
                ? t.Data
                : throw new KeyNotFoundException($"Whisper tensor '{name}' not found in the model.");
    }
}
