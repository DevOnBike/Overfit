// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
    /// <summary>
    /// Plain-span inference kernels for the Whisper encoder/decoder (no autograd, no graph). Correctness-first
    /// (TensorPrimitives.Dot for the matmuls; SIMD/fusion later). Shared by <see cref="WhisperEncoder"/> and
    /// (later) the decoder.
    /// </summary>
    internal static class WhisperKernels
    {
        private const float GeluC0 = 0.7978845608028654f; // sqrt(2/π)
        private const float GeluC1 = 0.044715f;

        /// <summary>GELU (tanh approximation, matching whisper.cpp's <c>ggml_gelu</c>), in place.</summary>
        public static void GeluInPlace(Span<float> x)
        {
            for (var i = 0; i < x.Length; i++)
            {
                var v = x[i];
                x[i] = 0.5f * v * (1f + MathF.Tanh(GeluC0 * (v + GeluC1 * v * v * v)));
            }
        }

        /// <summary>Row-wise LayerNorm: <c>out[t] = gamma · (x[t] − mean) / sqrt(var + eps) + beta</c>.
        /// <paramref name="x"/>/<paramref name="dst"/> are <c>[rows × dim]</c>.</summary>
        public static void LayerNorm(
            ReadOnlySpan<float> x, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta,
            Span<float> dst, int rows, int dim, float eps = 1e-5f)
        {
            for (var r = 0; r < rows; r++)
            {
                var row = x.Slice(r * dim, dim);
                var outRow = dst.Slice(r * dim, dim);
                var mean = 0f;
                for (var i = 0; i < dim; i++) { mean += row[i]; }
                mean /= dim;
                var variance = 0f;
                for (var i = 0; i < dim; i++) { var d = row[i] - mean; variance += d * d; }
                variance /= dim;
                var inv = 1f / MathF.Sqrt(variance + eps);
                for (var i = 0; i < dim; i++) { outRow[i] = (row[i] - mean) * inv * gamma[i] + beta[i]; }
            }
        }

        /// <summary>Linear: <c>out[t, o] = Σ_i x[t, i]·W[o, i] + b[o]</c>. <paramref name="weight"/> is
        /// output-major <c>[outDim × inDim]</c>; <paramref name="bias"/> may be empty (no bias).</summary>
        public static void Linear(
            ReadOnlySpan<float> x, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
            Span<float> dst, int rows, int inDim, int outDim)
        {
            for (var t = 0; t < rows; t++)
            {
                var xr = x.Slice(t * inDim, inDim);
                var outRow = dst.Slice(t * outDim, outDim);
                for (var o = 0; o < outDim; o++)
                {
                    var v = TensorPrimitives.Dot(xr, weight.Slice(o * inDim, inDim));
                    outRow[o] = bias.IsEmpty ? v : v + bias[o];
                }
            }
        }

        /// <summary>1-D convolution over time. <paramref name="input"/> is channel-major <c>[inC × tIn]</c>;
        /// <paramref name="weight"/> is <c>[outC × inC × kSize]</c>; output channel-major <c>[outC × tOut]</c>
        /// where <c>tOut = (tIn + 2·pad − kSize)/stride + 1</c>.</summary>
        public static void Conv1d(
            ReadOnlySpan<float> input, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
            Span<float> dst, int inC, int tIn, int outC, int kSize, int stride, int pad, int tOut)
        {
            for (var oc = 0; oc < outC; oc++)
            {
                var b = bias.IsEmpty ? 0f : bias[oc];
                for (var t = 0; t < tOut; t++)
                {
                    var acc = b;
                    var start = t * stride - pad;
                    for (var ic = 0; ic < inC; ic++)
                    {
                        var wBase = (oc * inC + ic) * kSize;
                        var inBase = ic * tIn;
                        for (var k = 0; k < kSize; k++)
                        {
                            var ti = start + k;
                            if (ti >= 0 && ti < tIn)
                            {
                                acc += weight[wBase + k] * input[inBase + ti];
                            }
                        }
                    }
                    dst[oc * tOut + t] = acc;
                }
            }
        }

        /// <summary>
        /// Multi-head attention. Query rows come from <paramref name="xq"/> <c>[tq × dModel]</c>; key/value
        /// rows from <paramref name="xkv"/> <c>[tkv × dModel]</c> (self-attention: same span; cross-attention:
        /// encoder output). Whisper has bias on Q/V/out but NOT on K. Non-causal (encoder + cross-attn) or
        /// causal (decoder self-attn). Writes <c>[tq × dModel]</c> into <paramref name="dst"/>.
        /// </summary>
        public static void MultiHeadAttention(
            ReadOnlySpan<float> xq, int tq, ReadOnlySpan<float> xkv, int tkv,
            int dModel, int nHeads,
            ReadOnlySpan<float> wq, ReadOnlySpan<float> bq,
            ReadOnlySpan<float> wk,
            ReadOnlySpan<float> wv, ReadOnlySpan<float> bv,
            ReadOnlySpan<float> wo, ReadOnlySpan<float> bo,
            Span<float> dst, bool causal)
        {
            var q = new float[tq * dModel];
            var k = new float[tkv * dModel];
            var v = new float[tkv * dModel];
            var attnOut = new float[tq * dModel];
            var scores = new float[tkv];
            MultiHeadAttention(xq, tq, xkv, tkv, dModel, nHeads, wq, bq, wk, wv, bv, wo, bo, dst, causal,
                q, k, v, attnOut, scores);
        }

        /// <summary>
        /// Allocation-free <see cref="MultiHeadAttention(System.ReadOnlySpan{float},int,System.ReadOnlySpan{float},int,int,int,System.ReadOnlySpan{float},System.ReadOnlySpan{float},System.ReadOnlySpan{float},System.ReadOnlySpan{float},System.ReadOnlySpan{float},System.ReadOnlySpan{float},System.ReadOnlySpan{float},System.Span{float},bool)"/>:
        /// the caller supplies the q/k/v/attnOut (each <c>[t × dModel]</c>) and scores (≥ tkv) scratch.
        /// </summary>
        public static void MultiHeadAttention(
            ReadOnlySpan<float> xq, int tq, ReadOnlySpan<float> xkv, int tkv,
            int dModel, int nHeads,
            ReadOnlySpan<float> wq, ReadOnlySpan<float> bq,
            ReadOnlySpan<float> wk,
            ReadOnlySpan<float> wv, ReadOnlySpan<float> bv,
            ReadOnlySpan<float> wo, ReadOnlySpan<float> bo,
            Span<float> dst, bool causal,
            Span<float> q, Span<float> k, Span<float> v, Span<float> attnOut, Span<float> scores)
        {
            var dHead = dModel / nHeads;
            var scale = 1f / MathF.Sqrt(dHead);

            Linear(xq, wq, bq, q, tq, dModel, dModel);
            Linear(xkv, wk, ReadOnlySpan<float>.Empty, k, tkv, dModel, dModel);
            Linear(xkv, wv, bv, v, tkv, dModel, dModel);

            for (var h = 0; h < nHeads; h++)
            {
                var off = h * dHead;
                for (var i = 0; i < tq; i++)
                {
                    var qi = q.Slice(i * dModel + off, dHead);
                    var valid = causal ? i + 1 : tkv;
                    var max = float.NegativeInfinity;
                    for (var j = 0; j < valid; j++)
                    {
                        var s = TensorPrimitives.Dot(qi, k.Slice(j * dModel + off, dHead)) * scale;
                        scores[j] = s;
                        if (s > max) { max = s; }
                    }
                    var sum = 0f;
                    for (var j = 0; j < valid; j++) { var e = MathF.Exp(scores[j] - max); scores[j] = e; sum += e; }
                    var inv = 1f / sum;

                    var outRow = attnOut.Slice(i * dModel + off, dHead);
                    outRow.Clear();
                    for (var j = 0; j < valid; j++)
                    {
                        TensorPrimitives.MultiplyAdd(v.Slice(j * dModel + off, dHead), scores[j] * inv, outRow, outRow);
                    }
                }
            }

            Linear(attnOut, wo, bo, dst, tq, dModel, dModel);
        }

        /// <summary>
        /// Single-query multi-head attention over a pre-projected K/V cache (the KV-cache decode path):
        /// query <paramref name="q"/> <c>[dModel]</c> attends over <paramref name="cacheK"/>/<paramref name="cacheV"/>
        /// (<c>[len × dModel]</c>, already projected). <paramref name="scores"/> is caller-owned scratch
        /// (≥ len). Writes the attended vector <c>[dModel]</c> into <paramref name="dst"/>. Zero-allocation.
        /// </summary>
        public static void SingleQueryAttention(
            ReadOnlySpan<float> q, ReadOnlySpan<float> cacheK, ReadOnlySpan<float> cacheV, int len,
            int dModel, int nHeads, Span<float> scores, Span<float> dst)
        {
            var dHead = dModel / nHeads;
            var scale = 1f / MathF.Sqrt(dHead);
            for (var h = 0; h < nHeads; h++)
            {
                var off = h * dHead;
                var qh = q.Slice(off, dHead);
                var max = float.NegativeInfinity;
                for (var j = 0; j < len; j++)
                {
                    var s = TensorPrimitives.Dot(qh, cacheK.Slice(j * dModel + off, dHead)) * scale;
                    scores[j] = s;
                    if (s > max) { max = s; }
                }
                var sum = 0f;
                for (var j = 0; j < len; j++) { var e = MathF.Exp(scores[j] - max); scores[j] = e; sum += e; }
                var inv = 1f / sum;

                var outH = dst.Slice(off, dHead);
                outH.Clear();
                for (var j = 0; j < len; j++)
                {
                    TensorPrimitives.MultiplyAdd(cacheV.Slice(j * dModel + off, dHead), scores[j] * inv, outH, outH);
                }
            }
        }
    }
}
