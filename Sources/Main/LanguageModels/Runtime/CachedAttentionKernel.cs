// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Intrinsics;

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
            float scale)
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

            for (var t = 0; t < sequenceLength; t++)
            {
                var key = keys.Slice(t * headDimension, headDimension);
                var score = Dot(query, key) * scale;

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

            if (sumExp <= 0f || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
            {
                output.Slice(0, headDimension).Clear();
                return;
            }

            output.Slice(0, headDimension).Clear();

            var invSum = 1f / sumExp;

            for (var t = 0; t < sequenceLength; t++)
            {
                var probability = scoreScratch[t] * invSum;
                var value = values.Slice(t * headDimension, headDimension);

                var d = 0;
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
            float scale)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(sequenceLength);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDimension);

            if (sequenceLength == 0)
            {
                output.Slice(0, headDimension).Clear();
                return;
            }

            var maxScore = float.NegativeInfinity;
            for (var t = 0; t < sequenceLength; t++)
            {
                var key = keysQ.Slice(t * headDimension, headDimension);
                var score = DotF32I8(query, key) * keyScales[t] * scale;
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
            if (Avx2.IsSupported && n >= 8)
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
            if (Avx2.IsSupported)
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

            if (Avx2.IsSupported && n >= 8)
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
