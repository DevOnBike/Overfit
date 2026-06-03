// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
    /// <summary>
    /// Whisper text decoder (inference, plain-span): token-embedding + positional-embedding → N pre-LN blocks,
    /// each = <b>causal</b> self-attention + <b>cross-attention</b> to the encoder output + GELU MLP → final
    /// LayerNorm → logits over the vocab (LM head is the tied token embedding). No KV-cache yet — each
    /// <see cref="Forward"/> recomputes the whole sequence (fine for short transcriptions; cache is a later
    /// optimization). <see cref="Decode"/> runs the greedy autoregressive loop.
    /// </summary>
    public sealed class WhisperDecoder
    {
        private readonly WhisperModel _m;
        private readonly int _nState;
        private readonly int _nHead;
        private readonly int _nLayer;
        private readonly int _nVocab;
        private readonly float[] _tokenEmbedding; // [nVocab, nState]

        public WhisperDecoder(WhisperModel model)
        {
            _m = model;
            _nState = model.Config.NTextState;
            _nHead = model.Config.NTextHead;
            _nLayer = model.Config.NTextLayer;
            _nVocab = model.Config.NVocab;
            _tokenEmbedding = T("decoder.token_embedding.weight");
        }

        /// <summary>
        /// Forward over <paramref name="tokens"/> attending the fixed encoder output
        /// (<paramref name="encoderOut"/> <c>[nCtx × nState]</c>); returns the next-token logits for the LAST
        /// position (<c>[nVocab]</c>).
        /// </summary>
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
                var next = ArgMax(logits);
                if (next == endOfTranscript) { break; }
                tokens.Add(next);
                produced.Add(next);
            }
            return produced.ToArray();
        }

        private static int ArgMax(ReadOnlySpan<float> v)
        {
            int best = 0; var bv = v[0];
            for (var i = 1; i < v.Length; i++) { if (v[i] > bv) { bv = v[i]; best = i; } }
            return best;
        }

        private float[] T(string name)
            => _m.Tensors.TryGetValue(name, out var t)
                ? t.Data
                : throw new KeyNotFoundException($"Whisper tensor '{name}' not found in the model.");
    }
}
