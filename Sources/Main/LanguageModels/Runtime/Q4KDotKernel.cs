// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.Runtime;

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
    /// <para>The AVX2 path feeds the Q4 nibbles straight into <c>vpmaddubsw</c>:
    /// they are unsigned [0,15] by construction, so they <i>are</i> the unsigned
    /// operand — no <c>vpsignb</c> trick needed (unlike the signed×signed Q8_0
    /// dot). The scalar fallback is bit-identical (pure INT32 arithmetic).</para>
    ///
    /// Parity-tested against the F32 decode of <see cref="GgmlDequant.DecodeQ4_KBlock"/>.
    /// </summary>
    public static unsafe class Q4KDotKernel
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

            // Main term: Σₛ scale[s]·intdotₛ over the 256 nibbles.
            var mainAcc = MainDot(q4kBlock.Slice(16, 128), q8Block, scales);

            // Min-correction: Σₛ min[s]·(sum of q8 over sub-block s). Each
            // sub-block spans two 16-element bsum groups. Tiny — kept scalar.
            var minAcc = 0;
            for (var s = 0; s < 8; s++)
            {
                minAcc += mins[s] * (bsums[2 * s] + bsums[2 * s + 1]);
            }

            return q8Scale * (d * mainAcc - dmin * minAcc);
        }

        /// <summary>
        /// <c>Σₛ scale[s]·intdotₛ</c> — the main accumulator. AVX2 unpacks the
        /// 4-bit nibbles and feeds them straight into <c>vpmaddubsw</c>; the
        /// scalar fallback is bit-identical (INT32 arithmetic throughout).
        /// </summary>
        private static int MainDot(ReadOnlySpan<byte> qs, ReadOnlySpan<sbyte> q8, ReadOnlySpan<byte> scales)
        {
            if (Avx2.IsSupported)
            {
                ref var qsRef = ref MemoryMarshal.GetReference(qs);
                ref var q8Ref = ref MemoryMarshal.GetReference(q8);
                var maskLow = Vector256.Create((byte)0x0F);

                var acc = 0;
                // Four 32-byte nibble runs; each carries two sub-blocks — the
                // low nibbles (even s) and the high nibbles (odd s).
                for (var p = 0; p < 4; p++)
                {
                    var packed = Vector256.LoadUnsafe(ref Unsafe.Add(ref qsRef, 32 * p));
                    var lowNibbles = Avx2.And(packed, maskLow);
                    var highNibbles = Avx2.And(
                        Avx2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), maskLow);

                    var q8Even = Vector256.LoadUnsafe(ref Unsafe.Add(ref q8Ref, 64 * p));
                    var q8Odd = Vector256.LoadUnsafe(ref Unsafe.Add(ref q8Ref, 64 * p + 32));

                    acc += scales[2 * p] * Int8BlockDot(lowNibbles, q8Even)
                         + scales[2 * p + 1] * Int8BlockDot(highNibbles, q8Odd);
                }
                return acc;
            }

            var sum = 0;
            for (var s = 0; s < 8; s++)
            {
                var qsBase = 32 * (s >> 1);
                var high = (s & 1) == 1;
                var q8Base = 32 * s;

                var dot = 0;
                for (var e = 0; e < 32; e++)
                {
                    var packed = qs[qsBase + e];
                    var nibble = high ? (packed >> 4) : (packed & 0x0F);
                    dot += nibble * q8[q8Base + e];
                }
                sum += scales[s] * dot;
            }
            return sum;
        }

        /// <summary>
        /// <c>Σ nibᵢ·q8ᵢ</c> over one 32-lane block as INT32. <c>vpmaddubsw</c>
        /// takes the unsigned nibbles × signed q8 directly — nibbles ∈ [0,15],
        /// q8 ∈ [-128,127], so the 16-bit pair sums (≤ 2·15·128 = 3840) never
        /// saturate; <c>vpmaddwd</c> then widens to INT32.
        /// </summary>
        private static int Int8BlockDot(Vector256<byte> nibbles, Vector256<sbyte> q8)
        {
            var pairs16 = Avx2.MultiplyAddAdjacent(nibbles, q8);
            var pairs32 = Avx2.MultiplyAddAdjacent(pairs16, Vector256.Create((short)1));
            return Vector256.Sum(pairs32);
        }

        /// <summary>
        /// Quantized single-token projection: <c>output = bias + input @ W</c>,
        /// W resident as Q4_K (output-major). The F32 <paramref name="input"/> is
        /// quantized once to Q8_K into the caller-owned scratch, then each output
        /// is the sum of <see cref="Dot"/> over its super-blocks. Sequential
        /// reference shape; the decode path uses <see cref="ProjectParallel"/>.
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

            ValidateProjectArguments(input, bias, output, inputSize, outputSize);

            QuantizeActivationQ8K(
                input.Slice(0, inputSize), activationQuants, activationScales, activationBsums);

            ProjectPreQuantized(weight, bias, output, activationQuants, activationScales, activationBsums);
        }

        /// <summary>
        /// Sequential projection from an **already-quantized** Q8_K activation —
        /// <see cref="Project"/> minus the quantize pass. Lets a caller quantize a
        /// shared input once (e.g. the attention <c>hidden</c> row, reused across
        /// every head's Q/K/V) and run several projections against it without
        /// re-quantizing. The scratch must already hold the Q8_K form of the
        /// projection's input (same layout <see cref="QuantizeActivationQ8K"/> writes).
        /// </summary>
        public static void ProjectPreQuantized(
            Q4KWeight weight,
            ReadOnlySpan<float> bias,
            Span<float> output,
            ReadOnlySpan<sbyte> activationQuants,
            ReadOnlySpan<float> activationScales,
            ReadOnlySpan<short> activationBsums)
        {
            ArgumentNullException.ThrowIfNull(weight);

            var outputSize = weight.OutputSize;
            var superBlocksPerRow = weight.SuperBlocksPerRow;

            var blocks = weight.BlockSpan;
            for (var o = 0; o < outputSize; o++)
            {
                var rowBase = (long)o * superBlocksPerRow * Q4KWeight.SuperBlockBytes;
                var sum = 0f;
                for (var sb = 0; sb < superBlocksPerRow; sb++)
                {
                    var block = blocks.Slice(
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

        /// <summary>
        /// Parallel quantized projection — <see cref="Project"/> with the output
        /// loop split across the zero-allocation <c>OverfitParallelFor</c> worker
        /// pool. The activation is quantized once (sequentially) into the
        /// caller-owned scratch, then each worker computes a disjoint band of
        /// output dots. Bit-identical to <see cref="Project"/>.
        /// </summary>
        public static void ProjectParallel(
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

            ValidateProjectArguments(input, bias, output, inputSize, outputSize);

            QuantizeActivationQ8K(
                input.Slice(0, inputSize), activationQuants, activationScales, activationBsums);

            fixed (byte* blocksPtr = weight.BlockSpan)
            fixed (sbyte* actQuants = activationQuants)
            fixed (float* actScales = activationScales)
            fixed (short* actBsums = activationBsums)
            fixed (float* biasPtr = bias)
            fixed (float* outputPtr = output)
            {
                var context = new Q4KProjectContext
                {
                    Blocks = blocksPtr,
                    ActivationQuants = actQuants,
                    ActivationScales = actScales,
                    ActivationBsums = actBsums,
                    Bias = biasPtr,
                    BiasLength = bias.Length,
                    Output = outputPtr,
                    SuperBlocksPerRow = weight.SuperBlocksPerRow,
                };

                OverfitParallelFor.For(0, outputSize, &ProjectChunk, &context);
            }
        }

        /// <summary>Worker body for <see cref="ProjectParallel"/> — one disjoint band of output rows.</summary>
        private static void ProjectChunk(int chunkStart, int chunkEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<Q4KProjectContext>(context);
            var superBlocksPerRow = ctx.SuperBlocksPerRow;

            for (var o = chunkStart; o < chunkEnd; o++)
            {
                var rowBase = (long)o * superBlocksPerRow * Q4KWeight.SuperBlockBytes;
                var sum = 0f;
                for (var sb = 0; sb < superBlocksPerRow; sb++)
                {
                    var block = new ReadOnlySpan<byte>(
                        ctx.Blocks + rowBase + (long)sb * Q4KWeight.SuperBlockBytes,
                        Q4KWeight.SuperBlockBytes);
                    var q8 = new ReadOnlySpan<sbyte>(
                        ctx.ActivationQuants + sb * SuperBlockElements, SuperBlockElements);
                    var bsums = new ReadOnlySpan<short>(
                        ctx.ActivationBsums + sb * GroupsPerSuperBlock, GroupsPerSuperBlock);

                    sum += Dot(block, q8, ctx.ActivationScales[sb], bsums);
                }

                ctx.Output[o] = ctx.BiasLength == 0 ? sum : ctx.Bias[o] + sum;
            }
        }

        private struct Q4KProjectContext
        {
            public byte* Blocks;
            public sbyte* ActivationQuants;
            public float* ActivationScales;
            public short* ActivationBsums;
            public float* Bias;
            public int BiasLength;
            public float* Output;
            public int SuperBlocksPerRow;
        }

        /// <summary>
        /// Batched Q4_K projection: <paramref name="rows"/> activation rows × one weight matrix — the
        /// prefill counterpart of <see cref="ProjectParallel"/>. Each row is quantized to Q8_K once into
        /// caller-owned scratch; the output loop is split across <c>OverfitParallelFor</c> with the
        /// <b>rows loop innermost</b>, so each weight output row's super-blocks are read from DRAM once
        /// and reused (cache-hot) across all <paramref name="rows"/> dots — cutting weight byte-traffic
        /// ~<paramref name="rows"/>× vs N× single-token <see cref="Project"/> (prefill is
        /// weight-bandwidth-bound). Bit-identical to N× <see cref="Project"/>.
        /// </summary>
        public static void ProjectBatched(
            ReadOnlySpan<float> input,
            int rows,
            Q4KWeight weight,
            ReadOnlySpan<float> bias,
            Span<float> output,
            Span<sbyte> activationQuants,
            Span<float> activationScales,
            Span<short> activationBsums)
        {
            ArgumentNullException.ThrowIfNull(weight);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);

            var inputSize = weight.InputSize;
            var outputSize = weight.OutputSize;
            var superBlocksPerRow = weight.SuperBlocksPerRow;
            var bsumsPerRow = superBlocksPerRow * GroupsPerSuperBlock;

            if (input.Length < (long)rows * inputSize)
            {
                throw new ArgumentException("Input span is smaller than rows * inputSize.", nameof(input));
            }
            if (output.Length < (long)rows * outputSize)
            {
                throw new ArgumentException("Output span is smaller than rows * outputSize.", nameof(output));
            }
            if (!bias.IsEmpty && bias.Length < outputSize)
            {
                throw new ArgumentException("Bias span is smaller than outputSize.", nameof(bias));
            }
            if (activationQuants.Length < (long)rows * inputSize
                || activationScales.Length < (long)rows * superBlocksPerRow
                || activationBsums.Length < (long)rows * bsumsPerRow)
            {
                throw new ArgumentException("Activation quantization scratch is too small for rows.");
            }

            for (var n = 0; n < rows; n++)
            {
                QuantizeActivationQ8K(
                    input.Slice(n * inputSize, inputSize),
                    activationQuants.Slice(n * inputSize, inputSize),
                    activationScales.Slice(n * superBlocksPerRow, superBlocksPerRow),
                    activationBsums.Slice(n * bsumsPerRow, bsumsPerRow));
            }

            fixed (byte* blocksPtr = weight.BlockSpan)
            fixed (sbyte* actQuants = activationQuants)
            fixed (float* actScales = activationScales)
            fixed (short* actBsums = activationBsums)
            fixed (float* biasPtr = bias)
            fixed (float* outputPtr = output)
            {
                var context = new Q4KBatchedContext
                {
                    Blocks = blocksPtr,
                    ActivationQuants = actQuants,
                    ActivationScales = actScales,
                    ActivationBsums = actBsums,
                    Bias = biasPtr,
                    BiasLength = bias.Length,
                    Output = outputPtr,
                    SuperBlocksPerRow = superBlocksPerRow,
                    InputSize = inputSize,
                    OutputSize = outputSize,
                    BsumsPerRow = bsumsPerRow,
                    Rows = rows,
                };

                OverfitParallelFor.For(0, outputSize, &ProjectBatchedChunk, &context);
            }
        }

        /// <summary>Worker for <see cref="ProjectBatched"/> — a band of output cols, rows innermost.</summary>
        private static void ProjectBatchedChunk(int chunkStart, int chunkEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<Q4KBatchedContext>(context);
            var superBlocksPerRow = ctx.SuperBlocksPerRow;

            for (var o = chunkStart; o < chunkEnd; o++)
            {
                var rowBase = (long)o * superBlocksPerRow * Q4KWeight.SuperBlockBytes;
                var biasO = ctx.BiasLength == 0 ? 0f : ctx.Bias[o];

                for (var n = 0; n < ctx.Rows; n++)
                {
                    var actQBase = (long)n * ctx.InputSize;
                    var actSBase = (long)n * superBlocksPerRow;
                    var actBBase = (long)n * ctx.BsumsPerRow;

                    var sum = 0f;
                    for (var sb = 0; sb < superBlocksPerRow; sb++)
                    {
                        var block = new ReadOnlySpan<byte>(
                            ctx.Blocks + rowBase + (long)sb * Q4KWeight.SuperBlockBytes,
                            Q4KWeight.SuperBlockBytes);
                        var q8 = new ReadOnlySpan<sbyte>(
                            ctx.ActivationQuants + actQBase + sb * SuperBlockElements, SuperBlockElements);
                        var bsums = new ReadOnlySpan<short>(
                            ctx.ActivationBsums + actBBase + sb * GroupsPerSuperBlock, GroupsPerSuperBlock);

                        sum += Dot(block, q8, ctx.ActivationScales[actSBase + sb], bsums);
                    }

                    ctx.Output[(long)n * ctx.OutputSize + o] = biasO + sum;
                }
            }
        }

        private struct Q4KBatchedContext
        {
            public byte* Blocks;
            public sbyte* ActivationQuants;
            public float* ActivationScales;
            public short* ActivationBsums;
            public float* Bias;
            public int BiasLength;
            public float* Output;
            public int SuperBlocksPerRow;
            public int InputSize;
            public int OutputSize;
            public int BsumsPerRow;
            public int Rows;
        }

        private static void ValidateProjectArguments(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
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
        }
    }
}
