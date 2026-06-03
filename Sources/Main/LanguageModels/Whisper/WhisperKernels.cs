// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
    /// <summary>
    /// Plain-span inference kernels for the Whisper encoder/decoder (no autograd, no graph). SIMD via
    /// TensorPrimitives; the large encoder matmuls/convs/attention are multi-threaded over independent output
    /// rows / channels / heads via <see cref="OverfitParallelFor"/> (above a work threshold, so the decoder's
    /// single-row calls stay sequential). Shared by <see cref="WhisperEncoder"/> and <see cref="WhisperDecoder"/>.
    /// </summary>
    internal static unsafe class WhisperKernels
    {
        // Parallelize a kernel only when total work (rows·in·out etc.) clears this; below it, dispatch overhead
        // dominates and the sequential SIMD path wins (matches LinearKernels' rationale).
        private const long ParallelThreshold = 1 << 18; // 262144
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
        /// output-major <c>[outDim × inDim]</c>; <paramref name="bias"/> may be empty (no bias). Parallelized
        /// over rows above <see cref="ParallelThreshold"/>.</summary>
        public static void Linear(
            ReadOnlySpan<float> x, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
            Span<float> dst, int rows, int inDim, int outDim)
        {
            if (rows <= 1 || (long)rows * inDim * outDim < ParallelThreshold)
            {
                for (var t = 0; t < rows; t++)
                {
                    LinearRow(x.Slice(t * inDim, inDim), weight, bias, dst.Slice(t * outDim, outDim), inDim, outDim);
                }
                return;
            }

            fixed (float* xp = x, wp = weight, bp = bias, dp = dst)
            {
                var ctx = new LinearCtx(xp, wp, bias.IsEmpty ? null : bp, dp, inDim, outDim);
                OverfitParallelFor.For(0, rows, &LinearWorker, &ctx);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void LinearRow(ReadOnlySpan<float> xr, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
            Span<float> outRow, int inDim, int outDim)
        {
            for (var o = 0; o < outDim; o++)
            {
                var v = TensorPrimitives.Dot(xr, weight.Slice(o * inDim, inDim));
                outRow[o] = bias.IsEmpty ? v : v + bias[o];
            }
        }

        private readonly struct LinearCtx
        {
            public readonly float* X, W, B, D;
            public readonly int InDim, OutDim;
            public LinearCtx(float* x, float* w, float* b, float* d, int inDim, int outDim)
            {
                X = x; W = w; B = b; D = d; InDim = inDim; OutDim = outDim;
            }
        }

        private static void LinearWorker(int start, int end, void* ctxPtr)
        {
            ref var c = ref Unsafe.AsRef<LinearCtx>(ctxPtr);
            var bias = c.B == null ? ReadOnlySpan<float>.Empty : new ReadOnlySpan<float>(c.B, c.OutDim);
            for (var t = start; t < end; t++)
            {
                LinearRow(new ReadOnlySpan<float>(c.X + (long)t * c.InDim, c.InDim), new ReadOnlySpan<float>(c.W, c.OutDim * c.InDim),
                    bias, new Span<float>(c.D + (long)t * c.OutDim, c.OutDim), c.InDim, c.OutDim);
            }
        }

        /// <summary>1-D convolution over time. <paramref name="input"/> is channel-major <c>[inC × tIn]</c>;
        /// <paramref name="weight"/> is <c>[outC × inC × kSize]</c>; output channel-major <c>[outC × tOut]</c>
        /// where <c>tOut = (tIn + 2·pad − kSize)/stride + 1</c>.</summary>
        public static void Conv1d(
            ReadOnlySpan<float> input, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
            Span<float> dst, int inC, int tIn, int outC, int kSize, int stride, int pad, int tOut)
        {
            if ((long)outC * tOut * inC * kSize < ParallelThreshold)
            {
                for (var oc = 0; oc < outC; oc++)
                {
                    Conv1dChannel(oc, input, weight, bias, dst, inC, tIn, kSize, stride, pad, tOut);
                }
                return;
            }

            fixed (float* ip = input, wp = weight, bp = bias, dp = dst)
            {
                var ctx = new Conv1dCtx(ip, wp, bias.IsEmpty ? null : bp, dp, inC, tIn, kSize, stride, pad, tOut);
                OverfitParallelFor.For(0, outC, &Conv1dWorker, &ctx);
            }
        }

        private static void Conv1dChannel(int oc, ReadOnlySpan<float> input, ReadOnlySpan<float> weight,
            ReadOnlySpan<float> bias, Span<float> dst, int inC, int tIn, int kSize, int stride, int pad, int tOut)
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

        private readonly struct Conv1dCtx
        {
            public readonly float* In, W, B, D;
            public readonly int InC, TIn, KSize, Stride, Pad, TOut;
            public Conv1dCtx(float* inp, float* w, float* b, float* d, int inC, int tIn, int kSize, int stride, int pad, int tOut)
            {
                In = inp; W = w; B = b; D = d; InC = inC; TIn = tIn; KSize = kSize; Stride = stride; Pad = pad; TOut = tOut;
            }
        }

        private static void Conv1dWorker(int start, int end, void* ctxPtr)
        {
            ref var c = ref Unsafe.AsRef<Conv1dCtx>(ctxPtr);
            // Spans cover up to index `end` (the chunk's exclusive upper bound); oc ∈ [start, end) only reads/writes below it.
            var input = new ReadOnlySpan<float>(c.In, c.InC * c.TIn);
            var weight = new ReadOnlySpan<float>(c.W, end * c.InC * c.KSize);
            var bias = c.B == null ? ReadOnlySpan<float>.Empty : new ReadOnlySpan<float>(c.B, end);
            var dst = new Span<float>(c.D, end * c.TOut);
            for (var oc = start; oc < end; oc++)
            {
                Conv1dChannel(oc, input, weight, bias, dst, c.InC, c.TIn, c.KSize, c.Stride, c.Pad, c.TOut);
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

            if ((long)nHeads * tq * tkv * dHead < ParallelThreshold)
            {
                for (var h = 0; h < nHeads; h++)
                {
                    MhaHead(h, q, k, v, attnOut, scores, tq, tkv, dModel, dHead, scale, causal);
                }
            }
            else
            {
                fixed (float* qp = q, kp = k, vp = v, ap = attnOut)
                {
                    var ctx = new MhaCtx(qp, kp, vp, ap, tq, tkv, dModel, dHead, scale, causal);
                    OverfitParallelFor.For(0, nHeads, &MhaWorker, &ctx);
                }
            }

            Linear(attnOut, wo, bo, dst, tq, dModel, dModel);
        }

        private static void MhaHead(int h, ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
            Span<float> attnOut, Span<float> scores, int tq, int tkv, int dModel, int dHead, float scale, bool causal)
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

        private readonly struct MhaCtx
        {
            public readonly float* Q, K, V, A;
            public readonly int Tq, Tkv, DModel, DHead;
            public readonly float Scale;
            public readonly bool Causal;
            public MhaCtx(float* q, float* k, float* v, float* a, int tq, int tkv, int dModel, int dHead, float scale, bool causal)
            {
                Q = q; K = k; V = v; A = a; Tq = tq; Tkv = tkv; DModel = dModel; DHead = dHead; Scale = scale; Causal = causal;
            }
        }

        private static void MhaWorker(int start, int end, void* ctxPtr)
        {
            ref var c = ref Unsafe.AsRef<MhaCtx>(ctxPtr);
            Span<float> scores = c.Tkv <= 8192 ? stackalloc float[c.Tkv] : new float[c.Tkv];
            var q = new ReadOnlySpan<float>(c.Q, c.Tq * c.DModel);
            var k = new ReadOnlySpan<float>(c.K, c.Tkv * c.DModel);
            var v = new ReadOnlySpan<float>(c.V, c.Tkv * c.DModel);
            var attnOut = new Span<float>(c.A, c.Tq * c.DModel);
            for (var h = start; h < end; h++)
            {
                MhaHead(h, q, k, v, attnOut, scores, c.Tq, c.Tkv, c.DModel, c.DHead, c.Scale, c.Causal);
            }
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
