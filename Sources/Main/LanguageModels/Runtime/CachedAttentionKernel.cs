// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Single-query cached attention kernel for autoregressive SLM decode.
    ///
    /// This kernel computes one attention head for one newly generated token:
    ///
    /// scores[t] = dot(query, key[t]) * scale
    /// probs = softmax(scores)
    /// output = sum_t(probs[t] * value[t])
    ///
    /// Expected layout:
    ///
    /// query:  [headDim]
    /// keys:   [sequenceLength, headDim]
    /// values: [sequenceLength, headDim]
    /// output: [headDim]
    /// scoreScratch: [sequenceLength]
    ///
    /// This is intentionally small and allocation-free. It does not own buffers,
    /// does not allocate scratch arrays and does not know about layers or heads.
    /// The caller supplies slices from KeyValueCache.
    /// </summary>
    public static class CachedAttentionKernel
    {
        public static void ComputeSingleHead(
            ReadOnlySpan<float> query,
            ReadOnlySpan<float> keys,
            ReadOnlySpan<float> values,
            Span<float> output,
            Span<float> scoreScratch,
            int sequenceLength,
            int headDimension,
            float scale,
            float softcap = 0f)
        {
            ValidateArguments(
                query,
                keys,
                values,
                output,
                scoreScratch,
                sequenceLength,
                headDimension);

            if (sequenceLength == 0)
            {
                output.Slice(0, headDimension).Clear();
                return;
            }

            var maxScore = float.NegativeInfinity;
            var invCap = softcap > 0f ? 1f / softcap : 0f;

            for (var t = 0; t < sequenceLength; t++)
            {
                var key = keys.Slice(t * headDimension, headDimension);

                // AblateScoreDot replaces the query·key GEMV with a cheap constant, to weigh the dot against
                // the exp below. Measurement only, never a production path.
                var score = (AblateScoreDot ? key[0] : Dot(query, key)) * scale;
                if (softcap > 0f) // Gemma-2 attn logit soft-cap: tanh(s/cap)·cap
                {
                    score = MathF.Tanh(score * invCap) * softcap;
                }

                scoreScratch[t] = score;

                if (score > maxScore)
                {
                    maxScore = score;
                }
            }

            var sumExp = 0.0f;

            if (UseVectorizedSoftmaxExp && !AblateSoftmaxExp)
            {
                // Vectorized softmax exp — the same lever SwiGLU already took (ApplySiLU): the scalar
                // per-element MathF.Exp was ~32% of attn_scores by ablation. scoreScratch[0..seqLen] is a
                // contiguous L1-resident buffer, exactly the bulk shape TensorPrimitives serves. Differs a few
                // ULP from scalar, so NOT byte-parity against the F32 reference — but prefill and decode both
                // reach this same method, so they stay bit-identical to EACH OTHER (the parity tests compare
                // the two paths, not against a stored scalar-exp value).
                var scores = scoreScratch.Slice(0, sequenceLength);
                TensorPrimitives.Subtract(scores, maxScore, scores);
                TensorPrimitives.Exp(scores, scores);
                sumExp = TensorPrimitives.Sum(scores);
            }

            if (!(UseVectorizedSoftmaxExp && !AblateSoftmaxExp))
            {
                for (var t = 0; t < sequenceLength; t++)
                {
                    var exp = AblateSoftmaxExp ? scoreScratch[t] - maxScore : MathF.Exp(scoreScratch[t] - maxScore);
                    scoreScratch[t] = exp;
                    sumExp += exp;
                }
            }

            if (sumExp <= 0f || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
            {
                output.Slice(0, headDimension).Clear();
                return;
            }

            output.Slice(0, headDimension).Clear();

            var invSum = 1f / sumExp;

            // Fold the normalisation into the probabilities once, so the inner loops below read a plain
            // coefficient. `scoreScratch[t] * invSum` computed here or there is the same product.
            for (var t = 0; t < sequenceLength; t++)
            {
                scoreScratch[t] *= invSum;
            }

            var dStart = 0;

            if (CpuFeatures.HasAvx2 && UseRegisterResidentValueSum)
            {
                dStart = AccumulateValuesBlocked(values, scoreScratch, output, sequenceLength, headDimension);
            }

            for (var t = 0; t < sequenceLength; t++)
            {
                var probability = scoreScratch[t];
                var value = values.Slice(t * headDimension, headDimension);

                var d = dStart;
                if (CpuFeatures.HasAvx2)
                {
                    // Vectorize over headDim. output[d] accumulates over t in ascending order
                    // (unchanged), and each d is independent + uses separate Multiply/Add (no FMA),
                    // so this is BIT-IDENTICAL to the scalar weighted sum.
                    ref var o = ref MemoryMarshal.GetReference(output);
                    ref var vv = ref MemoryMarshal.GetReference(value);
                    var probV = Vector256.Create(probability);
                    for (; d + 8 <= headDimension; d += 8)
                    {
                        var acc = Vector256.LoadUnsafe(ref o, (nuint)d);
                        var val = Vector256.LoadUnsafe(ref vv, (nuint)d);
                        Avx.Add(acc, Avx.Multiply(probV, val)).StoreUnsafe(ref o, (nuint)d);
                    }
                }

                for (; d < headDimension; d++)
                {
                    output[d] += probability * value[d];
                }
            }
        }

        /// <summary>A/B switch for <see cref="AccumulateValuesBlocked"/>; set <c>OVERFIT_ATTN_REGACC=0</c> to disable.</summary>
        internal static bool UseRegisterResidentValueSum =
            Environment.GetEnvironmentVariable(OverfitEnvironment.AttentionRegisterAccumulate) != "0";

        private static readonly string AblateMode = Environment.GetEnvironmentVariable(OverfitEnvironment.AttentionAblate) ?? "none";

        /// <summary>Measurement-only: replace the query·key dot with a constant, to size it against the exp.</summary>
        internal static bool AblateScoreDot = AblateMode is "dot" or "both";

        /// <summary>Measurement-only: skip the softmax exp, to size it against the query·key dot.</summary>
        internal static bool AblateSoftmaxExp = AblateMode is "exp" or "both";

        /// <summary>Vectorize the softmax exp via <c>TensorPrimitives</c>; set <c>OVERFIT_ATTN_VEXP=0</c> to disable.</summary>
        internal static bool UseVectorizedSoftmaxExp =
            Environment.GetEnvironmentVariable(OverfitEnvironment.AttentionVectorizedExp) != "0";

        /// <summary>
        /// The softmax-weighted value sum with the accumulators held in <b>registers across the whole
        /// <c>t</c> loop</b>, processing 64 output dimensions at a time. Returns the first dimension it did
        /// not cover, which the caller finishes with the original loop.
        ///
        /// <para><b>What it removes.</b> The straightforward order — for each <c>t</c>, walk every <c>d</c> —
        /// loads and stores the whole output accumulator once per <c>t</c>. Per <c>t</c> that is 512 B of value
        /// read against 512 B of accumulator read plus 512 B of accumulator write: <b>two thirds of the traffic
        /// is the accumulator going out to L1 and back</b>. Across a 672-token prefill head-layer that is
        /// ~226k iterations, ~347 MB moved of which ~231 MB is pure round-trip. Blocking <c>d</c> so the
        /// accumulators stay in registers reduces that to one load and one store per block per query.</para>
        ///
        /// <para>The value stream is unchanged in volume — the <c>d</c> blocks partition each value row, so the
        /// same bytes are read, just in two passes rather than one. Values fit L2 comfortably at these head
        /// dimensions.</para>
        ///
        /// <para><b>Bit-identical:</b> for every <c>d</c> the contributions are still summed in ascending
        /// <c>t</c> order, and the multiply and add stay separate — the no-FMA property the surrounding method
        /// documents is deliberate and preserved here.</para>
        /// </summary>
        private static int AccumulateValuesBlocked(
            ReadOnlySpan<float> values,
            ReadOnlySpan<float> probabilities,
            Span<float> output,
            int sequenceLength,
            int headDimension)
        {
            const int Lanes = 8;
            const int BlockWidth = 8 * Lanes;

            ref var o = ref MemoryMarshal.GetReference(output);
            ref var v = ref MemoryMarshal.GetReference(values);

            var d0 = 0;

            for (; d0 + BlockWidth <= headDimension; d0 += BlockWidth)
            {
                var a0 = Vector256<float>.Zero;
                var a1 = Vector256<float>.Zero;
                var a2 = Vector256<float>.Zero;
                var a3 = Vector256<float>.Zero;
                var a4 = Vector256<float>.Zero;
                var a5 = Vector256<float>.Zero;
                var a6 = Vector256<float>.Zero;
                var a7 = Vector256<float>.Zero;

                for (var t = 0; t < sequenceLength; t++)
                {
                    var probV = Vector256.Create(probabilities[t]);
                    var b = (nuint)((long)t * headDimension + d0);

                    a0 = Avx.Add(a0, Avx.Multiply(probV, Vector256.LoadUnsafe(ref v, b)));
                    a1 = Avx.Add(a1, Avx.Multiply(probV, Vector256.LoadUnsafe(ref v, b + 8)));
                    a2 = Avx.Add(a2, Avx.Multiply(probV, Vector256.LoadUnsafe(ref v, b + 16)));
                    a3 = Avx.Add(a3, Avx.Multiply(probV, Vector256.LoadUnsafe(ref v, b + 24)));
                    a4 = Avx.Add(a4, Avx.Multiply(probV, Vector256.LoadUnsafe(ref v, b + 32)));
                    a5 = Avx.Add(a5, Avx.Multiply(probV, Vector256.LoadUnsafe(ref v, b + 40)));
                    a6 = Avx.Add(a6, Avx.Multiply(probV, Vector256.LoadUnsafe(ref v, b + 48)));
                    a7 = Avx.Add(a7, Avx.Multiply(probV, Vector256.LoadUnsafe(ref v, b + 56)));
                }

                a0.StoreUnsafe(ref o, (nuint)d0);
                a1.StoreUnsafe(ref o, (nuint)(d0 + 8));
                a2.StoreUnsafe(ref o, (nuint)(d0 + 16));
                a3.StoreUnsafe(ref o, (nuint)(d0 + 24));
                a4.StoreUnsafe(ref o, (nuint)(d0 + 32));
                a5.StoreUnsafe(ref o, (nuint)(d0 + 40));
                a6.StoreUnsafe(ref o, (nuint)(d0 + 48));
                a7.StoreUnsafe(ref o, (nuint)(d0 + 56));
            }

            return d0;
        }

        /// <summary>
        /// Q8 KV-cache attend: identical math to <see cref="ComputeSingleHead"/> but K and V are
        /// resident as per-position symmetric int8 (one F32 scale per cached vector), so the
        /// attention read traffic drops ~4× — the long-context lever. Query stays F32 (precision +
        /// no per-token query-quant); each score dequantizes K on the fly
        /// (<c>score = scale · keyScale[t] · Σ q[d]·kQ[t][d]</c>) and the weighted-V sum dequantizes V
        /// (<c>out[d] += prob[t]·valueScale[t]·vQ[t][d]</c>). Not bit-identical to F32 (int8 round-trip),
        /// but cosine ≈ 1 — softmax + greedy decode are robust to it (validated in the bench).
        /// </summary>
        public static void ComputeSingleHeadQ8(
            ReadOnlySpan<float> query,
            ReadOnlySpan<sbyte> keysQ,
            ReadOnlySpan<float> keyScales,
            ReadOnlySpan<sbyte> valuesQ,
            ReadOnlySpan<float> valueScales,
            Span<float> output,
            Span<float> scoreScratch,
            int sequenceLength,
            int headDimension,
            float scale,
            float softcap = 0f)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(sequenceLength);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDimension);

            if (sequenceLength == 0)
            {
                output.Slice(0, headDimension).Clear();
                return;
            }

            var maxScore = float.NegativeInfinity;
            var invCap = softcap > 0f ? 1f / softcap : 0f;
            for (var t = 0; t < sequenceLength; t++)
            {
                var key = keysQ.Slice(t * headDimension, headDimension);
                var score = DotF32I8(query, key) * keyScales[t] * scale;
                if (softcap > 0f) // Gemma-2 attn logit soft-cap
                {
                    score = MathF.Tanh(score * invCap) * softcap;
                }
                scoreScratch[t] = score;
                if (score > maxScore)
                {
                    maxScore = score;
                }
            }

            var sumExp = 0.0f;
            for (var t = 0; t < sequenceLength; t++)
            {
                var exp = MathF.Exp(scoreScratch[t] - maxScore);
                scoreScratch[t] = exp;
                sumExp += exp;
            }

            output.Slice(0, headDimension).Clear();

            if (sumExp <= 0f || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
            {
                return;
            }

            var invSum = 1f / sumExp;
            for (var t = 0; t < sequenceLength; t++)
            {
                var coef = scoreScratch[t] * invSum * valueScales[t];
                var value = valuesQ.Slice(t * headDimension, headDimension);
                AxpyI8(coef, value, output, headDimension);
            }
        }

        /// <summary>F32·int8 dot — query stays F32, key bytes widened on the fly (two accumulators,
        /// reassociated like <see cref="Dot"/>; score precision is non-critical).</summary>
        private static float DotF32I8(ReadOnlySpan<float> q, ReadOnlySpan<sbyte> k)
        {
            var n = q.Length < k.Length ? q.Length : k.Length;
            var i = 0;
            var s = 0f;

            if (CpuFeatures.HasAvx2 && n >= 8)
            {
                ref var ql = ref MemoryMarshal.GetReference(q);
                ref var kl = ref MemoryMarshal.GetReference(k);
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                for (; i + 16 <= n; i += 16)
                {
                    var k0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Vector128.LoadUnsafe(ref kl, (nuint)i)));
                    var k1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Vector128.LoadUnsafe(ref kl, (nuint)(i + 8))));
                    acc0 = Avx.Add(acc0, Avx.Multiply(Vector256.LoadUnsafe(ref ql, (nuint)i), k0));
                    acc1 = Avx.Add(acc1, Avx.Multiply(Vector256.LoadUnsafe(ref ql, (nuint)(i + 8)), k1));
                }
                for (; i + 8 <= n; i += 8)
                {
                    var k0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Vector128.LoadUnsafe(ref kl, (nuint)i)));
                    acc0 = Avx.Add(acc0, Avx.Multiply(Vector256.LoadUnsafe(ref ql, (nuint)i), k0));
                }
                s = Vector256.Sum(Avx.Add(acc0, acc1));
            }
            for (; i < n; i++)
            {
                s += q[i] * k[i];
            }
            return s;
        }

        /// <summary>output[d] += coef · valueByte[d] (int8 widened), SIMD over headDim.</summary>
        private static void AxpyI8(float coef, ReadOnlySpan<sbyte> v, Span<float> output, int headDimension)
        {
            var d = 0;

            if (CpuFeatures.HasAvx2)
            {
                ref var o = ref MemoryMarshal.GetReference(output);
                ref var vv = ref MemoryMarshal.GetReference(v);
                var coefV = Vector256.Create(coef);

                for (; d + 8 <= headDimension; d += 8)
                {
                    var val = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Vector128.LoadUnsafe(ref vv, (nuint)d)));
                    var acc = Vector256.LoadUnsafe(ref o, (nuint)d);
                    Avx.Add(acc, Avx.Multiply(coefV, val)).StoreUnsafe(ref o, (nuint)d);
                }
            }
            for (; d < headDimension; d++)
            {
                output[d] += coef * v[d];
            }
        }

        public static void ComputeSingleHeadFromCache(
            IKeyValueCacheReader cache,
            int layerIndex,
            int headIndex,
            ReadOnlySpan<float> query,
            Span<float> output,
            Span<float> scoreScratch,
            float scale)
        {
            if (cache is null)
            {
                throw new ArgumentNullException(nameof(cache));
            }

            var sequenceLength = cache.CurrentLength;
            var headDimension = cache.HeadDimension;

            var keys = cache.GetKeyReadSpan(
                layerIndex,
                headIndex,
                fromPosition: 0,
                length: sequenceLength);

            var values = cache.GetValueReadSpan(
                layerIndex,
                headIndex,
                fromPosition: 0,
                length: sequenceLength);

            ComputeSingleHead(
                query,
                keys,
                values,
                output,
                scoreScratch,
                sequenceLength,
                headDimension,
                scale);
        }

        private static void ValidateArguments(
            ReadOnlySpan<float> query,
            ReadOnlySpan<float> keys,
            ReadOnlySpan<float> values,
            Span<float> output,
            Span<float> scoreScratch,
            int sequenceLength,
            int headDimension)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(sequenceLength);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDimension);

            if (query.Length < headDimension)
            {
                throw new ArgumentException("Query span is smaller than headDimension.", nameof(query));
            }

            if (output.Length < headDimension)
            {
                throw new ArgumentException("Output span is smaller than headDimension.", nameof(output));
            }

            if (scoreScratch.Length < sequenceLength)
            {
                throw new ArgumentException("Score scratch span is smaller than sequenceLength.", nameof(scoreScratch));
            }

            var requiredKvLength = sequenceLength * headDimension;

            if (keys.Length < requiredKvLength)
            {
                throw new ArgumentException("Keys span is smaller than sequenceLength * headDimension.", nameof(keys));
            }

            if (values.Length < requiredKvLength)
            {
                throw new ArgumentException("Values span is smaller than sequenceLength * headDimension.", nameof(values));
            }
        }

        private static float Dot(
            ReadOnlySpan<float> left,
            ReadOnlySpan<float> right)
        {
            var n = left.Length;

            if (CpuFeatures.HasAvx2 && n >= 8)
            {
                // Two accumulators break the scalar sum's loop-carried dependency (the latency
                // bottleneck of the score dot at long context). NOT bit-identical to the scalar
                // sequential sum (vectorized + reassociated) — it's marginally MORE accurate, like
                // llama.cpp's dot; greedy decode stays coherent. Score precision here is non-critical
                // (softmax is robust to ~1 ULP).
                ref var l = ref MemoryMarshal.GetReference(left);
                ref var r = ref MemoryMarshal.GetReference(right);

                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                var i = 0;
                for (; i + 16 <= n; i += 16)
                {
                    acc0 = Avx.Add(acc0, Avx.Multiply(
                        Vector256.LoadUnsafe(ref l, (nuint)i), Vector256.LoadUnsafe(ref r, (nuint)i)));
                    acc1 = Avx.Add(acc1, Avx.Multiply(
                        Vector256.LoadUnsafe(ref l, (nuint)(i + 8)), Vector256.LoadUnsafe(ref r, (nuint)(i + 8))));
                }
                for (; i + 8 <= n; i += 8)
                {
                    acc0 = Avx.Add(acc0, Avx.Multiply(
                        Vector256.LoadUnsafe(ref l, (nuint)i), Vector256.LoadUnsafe(ref r, (nuint)i)));
                }

                var sum = Vector256.Sum(Avx.Add(acc0, acc1));
                for (; i < n; i++)
                {
                    sum += left[i] * right[i];
                }

                return sum;
            }

            var s = 0f;
            for (var i = 0; i < n; i++)
            {
                s += left[i] * right[i];
            }

            return s;
        }
    }

}
