// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// F32 → K-quant <b>encoder</b> (the inverse of <see cref="GgmlDequant"/>). The only place in
    /// Overfit that writes quantized bytes — all other quant handling is read-only. Used to FREEZE
    /// an F32 weight as a 4-bit base for QLoRA-style fine-tuning (frozen quantized base + trainable
    /// LoRA adapter), keeping the base ~7× smaller in RAM during training.
    ///
    /// <para>Q4_K uses asymmetric per-sub-block (32-element) min/max quantization with 6-bit
    /// sub-block scales/mins under two FP16 super-scales — the exact layout
    /// <see cref="GgmlDequant.DecodeQ4_KBlock"/> reads back. This is the plain min/max scheme
    /// (ggml's reference adds an iterative RMSE refinement); it round-trips within Q4_K precision,
    /// which is all the frozen-base path needs.</para>
    /// </summary>
    public static class GgmlQuant
    {
        private const int SuperBlockElements = 256;
        private const int Q4_K_BlockBytes = 144;

        /// <summary>
        /// Encodes 256 F32 values into one Q4_K super-block (144 bytes), byte-compatible with
        /// <see cref="GgmlDequant.DecodeQ4_KBlock"/>.
        /// </summary>
        public static void EncodeQ4_KBlock(ReadOnlySpan<float> src256, Span<byte> block144)
        {
            if (src256.Length != SuperBlockElements)
            {
                throw new ArgumentException($"src must be exactly {SuperBlockElements} floats.", nameof(src256));
            }
            if (block144.Length != Q4_K_BlockBytes)
            {
                throw new ArgumentException($"block must be exactly {Q4_K_BlockBytes} bytes.", nameof(block144));
            }

            // 1. Per sub-block (8 × 32): float scale (≥0) and offset min (≥0), so x ≈ scale·q − min.
            Span<float> subScale = stackalloc float[8];
            Span<float> subMin = stackalloc float[8];
            for (var j = 0; j < 8; j++)
            {
                var lo = float.PositiveInfinity;
                var hi = float.NegativeInfinity;
                for (var e = 0; e < 32; e++)
                {
                    var v = src256[j * 32 + e];
                    if (v < lo)
                    {
                        lo = v;
                    }
                    if (v > hi)
                    {
                        hi = v;
                    }
                }

                // The reconstruction y = scale·q − min spans [−min, 15·scale − min]; the offset can
                // only be ≥ 0, so the smallest representable value is ≤ 0 → clamp lo to ≤ 0.
                if (lo > 0f)
                {
                    lo = 0f;
                }
                subScale[j] = (hi - lo) / 15f;
                subMin[j] = -lo;
            }

            // 2. Quantize the 8 scales + 8 mins to 6 bits under two FP16 super-scales (d, dmin).
            var maxScale = 0f;
            var maxMin = 0f;
            for (var j = 0; j < 8; j++)
            {
                if (subScale[j] > maxScale)
                {
                    maxScale = subScale[j];
                }
                if (subMin[j] > maxMin)
                {
                    maxMin = subMin[j];
                }
            }

            var invScale = maxScale > 0f ? 63f / maxScale : 0f;
            var invMin = maxMin > 0f ? 63f / maxMin : 0f;

            Span<byte> sc = stackalloc byte[8];
            Span<byte> m = stackalloc byte[8];
            for (var j = 0; j < 8; j++)
            {
                sc[j] = (byte)Math.Min(63, (int)MathF.Round(invScale * subScale[j]));
                m[j] = (byte)Math.Min(63, (int)MathF.Round(invMin * subMin[j]));
            }

            var d = maxScale / 63f;
            var dmin = maxMin / 63f;

            BinaryPrimitives.WriteUInt16LittleEndian(block144[..2], BitConverter.HalfToUInt16Bits((Half)d));
            BinaryPrimitives.WriteUInt16LittleEndian(block144.Slice(2, 2), BitConverter.HalfToUInt16Bits((Half)dmin));
            PackScalesMins(sc, m, block144.Slice(4, 12));

            // 3. Re-quantize the 256 nibbles with the QUANTIZED scales (so encode error matches decode).
            Span<byte> q = stackalloc byte[256];
            for (var j = 0; j < 8; j++)
            {
                var es = d * sc[j];
                var em = dmin * m[j];
                for (var e = 0; e < 32; e++)
                {
                    var i = j * 32 + e;
                    var l = es > 0f ? (int)MathF.Round((src256[i] + em) / es) : 0;
                    q[i] = (byte)Math.Clamp(l, 0, 15);
                }
            }

            // 4. Pack nibbles: qs[32p+e] = q[64p+e] | (q[64p+32+e] << 4)  (low = sub 2p, high = sub 2p+1).
            var qs = block144.Slice(16, 128);
            for (var p = 0; p < 4; p++)
            {
                for (var e = 0; e < 32; e++)
                {
                    qs[32 * p + e] = (byte)(q[64 * p + e] | (q[64 * p + 32 + e] << 4));
                }
            }
        }

        /// <summary>
        /// Quantizes a full <b>output-major</b> F32 weight matrix (<paramref name="outputSize"/> rows ×
        /// <paramref name="inputSize"/> cols, row-major) into Q4_K super-block bytes laid out exactly as
        /// <c>Q4KWeight</c> expects. <paramref name="inputSize"/> must be a multiple of 256.
        /// </summary>
        public static byte[] QuantizeQ4_K(ReadOnlySpan<float> f32, int inputSize, int outputSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputSize);
            if (inputSize % SuperBlockElements != 0)
            {
                throw new ArgumentException($"inputSize ({inputSize}) must be a multiple of {SuperBlockElements}.", nameof(inputSize));
            }
            if (f32.Length < (long)outputSize * inputSize)
            {
                throw new ArgumentException("f32 span smaller than outputSize * inputSize.", nameof(f32));
            }

            var sbPerRow = inputSize / SuperBlockElements;
            var bytes = new byte[(long)outputSize * sbPerRow * Q4_K_BlockBytes];
            for (var o = 0; o < outputSize; o++)
            {
                for (var sb = 0; sb < sbPerRow; sb++)
                {
                    var src = f32.Slice(o * inputSize + sb * SuperBlockElements, SuperBlockElements);
                    var dstOff = (o * sbPerRow + sb) * Q4_K_BlockBytes;
                    EncodeQ4_KBlock(src, bytes.AsSpan(dstOff, Q4_K_BlockBytes));
                }
            }

            return bytes;
        }

        // Inverse of GgmlDequant.UnpackQ4_KScalesMins — packs sc[8] + m[8] (6-bit) into 12 bytes.
        private static void PackScalesMins(ReadOnlySpan<byte> sc, ReadOnlySpan<byte> m, Span<byte> packed12)
        {
            for (var j = 0; j < 4; j++)
            {
                packed12[j] = (byte)((sc[j] & 0x3F) | ((sc[j + 4] >> 4) << 6));
                packed12[j + 4] = (byte)((m[j] & 0x3F) | ((m[j + 4] >> 4) << 6));
                packed12[j + 8] = (byte)((sc[j + 4] & 0x0F) | ((m[j + 4] & 0x0F) << 4));
            }
        }
    }
}
