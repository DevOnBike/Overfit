// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Q4_K × Q8_K quantized dot product — the decode primitive for Q4_K-resident
    /// weights (step 3 of the decode-kernel plan). The F32 activation is
    /// quantized to Q8_K (one 256-element super-block = one F32 scale + 256 int8
    /// quants + 16 int16 group sums); the dot mirrors llama.cpp's
    /// <c>ggml_vec_dot_q4_K_q8_K</c>:
    ///
    ///   Σ wᵢ·aᵢ  =  q8d · ( d · Σₛ scale[s]·intdotₛ  −  dmin · Σₛ min[s]·bsumpairₛ )
    ///
    /// where the 256 weights split into 8 sub-blocks of 32; <c>intdotₛ</c> is the
    /// integer dot of sub-block s's 4-bit weight nibbles with the q8 activations,
    /// and <c>bsumpairₛ</c> is the sum of those q8 activations (read from the
    /// Q8_K group sums). This identity is exact given the Q8_K activation — the
    /// only approximation is the activation quantization itself.
    ///
    /// This is the scalar reference shape; an AVX2 INT8-SIMD path is a later
    /// sub-step (3.1b). Standalone and parity-tested against the F32 decode of
    /// <see cref="GgmlDequant.DecodeQ4_KBlock"/>.
    /// </summary>
    public static class Q4KDotKernel
    {
        /// <summary>Elements per Q4_K / Q8_K super-block.</summary>
        public const int SuperBlockElements = 256;

        /// <summary>Activation group size for the Q8_K block sums (<c>bsums</c>).</summary>
        public const int GroupSize = 16;

        /// <summary>Q8_K group sums per super-block: 256 / 16.</summary>
        public const int GroupsPerSuperBlock = SuperBlockElements / GroupSize;

        /// <summary>
        /// Quantizes an F32 activation (length a multiple of
        /// <see cref="SuperBlockElements"/>) to Q8_K: a per-super-block F32 scale
        /// (<c>absmax/127</c>), 256 int8 quants per block, and 16 int16 group
        /// sums per block (each the sum of a <see cref="GroupSize"/>-element run
        /// of quants — the <c>bsums</c> the Q4_K min-correction term needs).
        /// </summary>
        public static void QuantizeActivationQ8K(
            ReadOnlySpan<float> source,
            Span<sbyte> quants,
            Span<float> blockScales,
            Span<short> bsums)
        {
            if (source.Length % SuperBlockElements != 0)
            {
                throw new ArgumentException(
                    $"Source length ({source.Length}) must be a multiple of {SuperBlockElements}.",
                    nameof(source));
            }

            var blocks = source.Length / SuperBlockElements;

            if (quants.Length < source.Length)
            {
                throw new ArgumentException("Quants span is smaller than source.", nameof(quants));
            }
            if (blockScales.Length < blocks)
            {
                throw new ArgumentException("Block-scales span is smaller than the block count.", nameof(blockScales));
            }
            if (bsums.Length < blocks * GroupsPerSuperBlock)
            {
                throw new ArgumentException("Bsums span is smaller than blocks * groups.", nameof(bsums));
            }

            for (var b = 0; b < blocks; b++)
            {
                var block = source.Slice(b * SuperBlockElements, SuperBlockElements);

                var absMax = 0f;
                for (var i = 0; i < SuperBlockElements; i++)
                {
                    var a = MathF.Abs(block[i]);
                    if (a > absMax)
                    {
                        absMax = a;
                    }
                }

                var scale = absMax / 127f;
                blockScales[b] = scale;

                var inverse = scale > 0f ? 1f / scale : 0f;
                for (var g = 0; g < GroupsPerSuperBlock; g++)
                {
                    var sum = 0;
                    for (var k = 0; k < GroupSize; k++)
                    {
                        var idx = g * GroupSize + k;
                        var q = (int)Math.Clamp(MathF.Round(block[idx] * inverse), -127f, 127f);
                        quants[b * SuperBlockElements + idx] = (sbyte)q;
                        sum += q;
                    }
                    bsums[b * GroupsPerSuperBlock + g] = (short)sum;
                }
            }
        }

        /// <summary>
        /// Contracts one Q4_K weight super-block (144 bytes) with one Q8_K
        /// activation super-block — 256 elements. Exact given the activation
        /// quantization; see the class remarks for the identity.
        /// </summary>
        public static float Dot(
            ReadOnlySpan<byte> q4kBlock,
            ReadOnlySpan<sbyte> q8Block,
            float q8Scale,
            ReadOnlySpan<short> bsums)
        {
            if (q4kBlock.Length < Q4KWeight.SuperBlockBytes)
            {
                throw new ArgumentException("Q4_K block span is smaller than 144 bytes.", nameof(q4kBlock));
            }
            if (q8Block.Length < SuperBlockElements)
            {
                throw new ArgumentException("Q8 block span is smaller than 256.", nameof(q8Block));
            }
            if (bsums.Length < GroupsPerSuperBlock)
            {
                throw new ArgumentException("Bsums span is smaller than 16.", nameof(bsums));
            }

            var d = (float)BitConverter.UInt16BitsToHalf(
                BinaryPrimitives.ReadUInt16LittleEndian(q4kBlock[..2]));
            var dmin = (float)BitConverter.UInt16BitsToHalf(
                BinaryPrimitives.ReadUInt16LittleEndian(q4kBlock.Slice(2, 2)));

            Span<byte> scales = stackalloc byte[8];
            Span<byte> mins = stackalloc byte[8];
            GgmlDequant.UnpackQ4_KScalesMins(q4kBlock.Slice(4, 12), scales, mins);

            var qs = q4kBlock.Slice(16, 128);   // 256 nibbles

            // mainAcc = Σₛ scale[s]·intdotₛ ; minAcc = Σₛ min[s]·(bsum of sub-block s).
            // Both stay INT32: scale,min ≤ 63; intdotₛ ≤ 32·15·127; the 8-term
            // sums are well under 2³¹.
            var mainAcc = 0;
            var minAcc = 0;
            for (var s = 0; s < 8; s++)
            {
                var qsBase = 32 * (s >> 1);       // two sub-blocks share a 32-byte nibble run
                var high = (s & 1) == 1;          // even s → low nibbles, odd s → high nibbles
                var q8Base = 32 * s;

                var dot = 0;
                for (var e = 0; e < 32; e++)
                {
                    var packed = qs[qsBase + e];
                    var nibble = high ? (packed >> 4) : (packed & 0x0F);
                    dot += nibble * q8Block[q8Base + e];
                }

                mainAcc += scales[s] * dot;
                minAcc += mins[s] * (bsums[2 * s] + bsums[2 * s + 1]);
            }

            return q8Scale * (d * mainAcc - dmin * minAcc);
        }

        /// <summary>
        /// Quantized single-token projection: <c>output = bias + input @ W</c>,
        /// W resident as Q4_K (output-major). The F32 <paramref name="input"/> is
        /// quantized once to Q8_K into the caller-owned scratch, then each output
        /// is the sum of <see cref="Dot"/> over its super-blocks. Sequential
        /// reference shape; a parallel path follows with the AVX2 kernel.
        /// </summary>
        public static void Project(
            ReadOnlySpan<float> input,
            Q4KWeight weight,
            ReadOnlySpan<float> bias,
            Span<float> output,
            Span<sbyte> activationQuants,
            Span<float> activationScales,
            Span<short> activationBsums)
        {
            ArgumentNullException.ThrowIfNull(weight);

            var inputSize = weight.InputSize;
            var outputSize = weight.OutputSize;
            var superBlocksPerRow = weight.SuperBlocksPerRow;

            if (input.Length < inputSize)
            {
                throw new ArgumentException("Input span is smaller than the weight's input size.", nameof(input));
            }
            if (output.Length < outputSize)
            {
                throw new ArgumentException("Output span is smaller than the weight's output size.", nameof(output));
            }
            if (!bias.IsEmpty && bias.Length < outputSize)
            {
                throw new ArgumentException("Bias span is smaller than outputSize.", nameof(bias));
            }

            QuantizeActivationQ8K(
                input.Slice(0, inputSize), activationQuants, activationScales, activationBsums);

            var blocks = weight.Blocks;
            for (var o = 0; o < outputSize; o++)
            {
                var rowBase = (long)o * superBlocksPerRow * Q4KWeight.SuperBlockBytes;
                var sum = 0f;
                for (var sb = 0; sb < superBlocksPerRow; sb++)
                {
                    var block = blocks.AsSpan(
                        (int)(rowBase + (long)sb * Q4KWeight.SuperBlockBytes), Q4KWeight.SuperBlockBytes);
                    sum += Dot(
                        block,
                        activationQuants.Slice(sb * SuperBlockElements, SuperBlockElements),
                        activationScales[sb],
                        activationBsums.Slice(sb * GroupsPerSuperBlock, GroupsPerSuperBlock));
                }

                output[o] = bias.IsEmpty ? sum : bias[o] + sum;
            }
        }
    }
}
