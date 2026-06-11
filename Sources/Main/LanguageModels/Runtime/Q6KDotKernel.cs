// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Q6_K × Q8_K quantized dot product — the decode primitive for Q6_K-resident
    /// weights (step 3.3). The F32 activation is quantized to Q8_K (the same
    /// shape Q4_K uses: per-256 F32 scale + 256 int8 quants + 16 int16 group
    /// sums); the dot expands the Q6_K layout against it.
    ///
    /// <para>
    /// The mathematical identity used by both the scalar and AVX2 paths:
    ///   Σ wᵢ·aᵢ = d · q8d · ( Σₛ scales[s] · unsignedDotₛ  −  32 · Σₛ scales[s] · bsums[s] )
    /// where <c>unsignedDotₛ = Σ_{i ∈ s} q[i] · q8[i]</c> over the UNSIGNED 6-bit
    /// quants q ∈ [0, 63]. The <c>−32</c> bias factors out as a scalar correction
    /// using the Q8_K group sums, freeing the inner loop to feed unsigned quants
    /// straight into <c>vpmaddubsw</c> (no sign trick needed, like Q4_K).
    /// </para>
    ///
    /// <para>
    /// Sub-block layout: 16 sub-blocks of 16 elements (vs Q4_K's 8 sub-blocks of
    /// 32). Each sub-block has a <i>signed</i> int8 scale (vs Q4_K's 6-bit
    /// unsigned). Q6_K has no min term.
    /// </para>
    ///
    /// <para>
    /// AVX2 path: 8 quant-groups of 32 elements per super-block, each group
    /// spans two sub-blocks (16+16 elements). One <c>vpmaddubsw</c> +
    /// <c>vpmaddwd</c> produces 8 int32 pair-sums; the lower four (sub-block A)
    /// and upper four (sub-block B) are summed and scaled separately, then
    /// accumulated. Bit-identical to the scalar fallback (pure INT32 arithmetic).
    /// </para>
    ///
    /// Parity-tested against the F32 decode of <see cref="GgmlDequant.DecodeQ6_KBlock"/>.
    /// </summary>
    public static unsafe class Q6KDotKernel
    {
        /// <summary>Elements per Q6_K / Q8_K super-block.</summary>
        public const int SuperBlockElements = 256;

        /// <summary>Q6_K sub-block size (also Q8_K's bsum group size).</summary>
        public const int GroupSize = 16;

        /// <summary>Q8_K group sums per super-block: 256 / 16.</summary>
        public const int GroupsPerSuperBlock = SuperBlockElements / GroupSize;

        /// <summary>
        /// Q8_K activation quantization — the same format Q4_K uses (per-256
        /// F32 scale + 256 int8 quants + 16 int16 group sums). Reuses
        /// <see cref="Q4KDotKernel.QuantizeActivationQ8K"/>; both K-quant kernels
        /// consume Q8_K-quantized activations and the format is shared.
        /// </summary>
        public static void QuantizeActivationQ8K(
            ReadOnlySpan<float> source,
            Span<sbyte> quants,
            Span<float> blockScales,
            Span<short> bsums)
            => Q4KDotKernel.QuantizeActivationQ8K(source, quants, blockScales, bsums);

        /// <summary>
        /// Contracts one Q6_K weight super-block (210 bytes) with one Q8_K
        /// activation super-block — 256 elements. Exact given the activation
        /// quantization; see the class remarks for the identity.
        /// </summary>
        public static float Dot(
            ReadOnlySpan<byte> q6kBlock,
            ReadOnlySpan<sbyte> q8Block,
            float q8Scale,
            ReadOnlySpan<short> bsums)
        {
            if (q6kBlock.Length < Q6KWeight.SuperBlockBytes)
            {
                throw new ArgumentException("Q6_K block span is smaller than 210 bytes.", nameof(q6kBlock));
            }
            if (q8Block.Length < SuperBlockElements)
            {
                throw new ArgumentException("Q8 block span is smaller than 256.", nameof(q8Block));
            }
            if (bsums.Length < GroupsPerSuperBlock)
            {
                throw new ArgumentException("Bsums span is smaller than 16.", nameof(bsums));
            }

            var ql = q6kBlock.Slice(0, 128);
            var qh = q6kBlock.Slice(128, 64);
            var sc = q6kBlock.Slice(192, 16);   // int8 SIGNED — sign-extend at use.
            var d = (float)BitConverter.UInt16BitsToHalf(
                BinaryPrimitives.ReadUInt16LittleEndian(q6kBlock.Slice(208, 2)));

            // Σₛ scales[s] · unsignedDotₛ — AVX2 unpacks the 6-bit quants and
            // feeds them straight into vpmaddubsw (q ∈ [0,63] is the unsigned
            // operand; q8 is signed). Scalar fallback is bit-identical.
            var unsignedTerm = UnsignedDotSuperBlock(ql, qh, q8Block, sc);

            // 32 · Σₛ scales[s] · bsums[s] — the bias correction. Tiny — scalar.
            var bsumTerm = 0;
            for (var s = 0; s < 16; s++)
            {
                bsumTerm += (sbyte)sc[s] * bsums[s];
            }

            var mainAcc = unsignedTerm - 32 * bsumTerm;
            return d * q8Scale * mainAcc;
        }

        /// <summary>
        /// <c>Σₛ scales[s] · unsignedDotₛ</c> across all 16 sub-blocks. AVX2
        /// processes 8 quant-groups of 32 elements (each group covers 2 sub-blocks
        /// of 16); the scalar fallback is bit-identical (INT32 arithmetic).
        /// </summary>
        private static int UnsignedDotSuperBlock(
            ReadOnlySpan<byte> ql,
            ReadOnlySpan<byte> qh,
            ReadOnlySpan<sbyte> q8Block,
            ReadOnlySpan<byte> scales)
        {
            if (CpuFeatures.HasAvx2)
            {
                ref var qlRef = ref MemoryMarshal.GetReference(ql);
                ref var qhRef = ref MemoryMarshal.GetReference(qh);
                ref var q8Ref = ref MemoryMarshal.GetReference(q8Block);
                var maskLow = Vector256.Create((byte)0x0F);
                var mask2bit = Vector256.Create((byte)0x03);
                var ones16 = Vector256.Create((short)1);

                var unsignedTerm = 0;
                for (var h = 0; h < 2; h++)
                {
                    var qlBase = 64 * h;
                    var qhBase = 32 * h;
                    var scBase = 8 * h;
                    var q8Base = 128 * h;

                    // ql[64 B] split into low/high nibbles; qh[32 B] split into
                    // four 2-bit fields by shift {0, 2, 4, 6}.
                    var qlA = Vector256.LoadUnsafe(ref Unsafe.Add(ref qlRef, qlBase));
                    var qlB = Vector256.LoadUnsafe(ref Unsafe.Add(ref qlRef, qlBase + 32));
                    var qhV = Vector256.LoadUnsafe(ref Unsafe.Add(ref qhRef, qhBase));

                    var qlALow = Avx2.And(qlA, maskLow);
                    var qlAHigh = Avx2.And(Avx2.ShiftRightLogical(qlA.AsUInt16(), 4).AsByte(), maskLow);
                    var qlBLow = Avx2.And(qlB, maskLow);
                    var qlBHigh = Avx2.And(Avx2.ShiftRightLogical(qlB.AsUInt16(), 4).AsByte(), maskLow);

                    var qhG0 = Avx2.ShiftLeftLogical(Avx2.And(qhV, mask2bit).AsUInt16(), 4).AsByte();
                    var qhG1 = Avx2.ShiftLeftLogical(
                        Avx2.And(Avx2.ShiftRightLogical(qhV.AsUInt16(), 2).AsByte(), mask2bit).AsUInt16(), 4).AsByte();
                    var qhG2 = Avx2.ShiftLeftLogical(
                        Avx2.And(Avx2.ShiftRightLogical(qhV.AsUInt16(), 4).AsByte(), mask2bit).AsUInt16(), 4).AsByte();
                    var qhG3 = Avx2.ShiftLeftLogical(
                        Avx2.And(Avx2.ShiftRightLogical(qhV.AsUInt16(), 6).AsByte(), mask2bit).AsUInt16(), 4).AsByte();

                    // Group g (∈ 0..3 within half) covers dst[g*32 .. g*32+32] and
                    // straddles sub-blocks scBase+2g (first 16) and scBase+2g+1
                    // (next 16). Matches DecodeQ6_KBlock's element layout.
                    var qG0 = Avx2.Or(qlALow, qhG0);
                    var qG1 = Avx2.Or(qlBLow, qhG1);
                    var qG2 = Avx2.Or(qlAHigh, qhG2);
                    var qG3 = Avx2.Or(qlBHigh, qhG3);

                    var q8G0 = Vector256.LoadUnsafe(ref Unsafe.Add(ref q8Ref, q8Base));
                    var q8G1 = Vector256.LoadUnsafe(ref Unsafe.Add(ref q8Ref, q8Base + 32));
                    var q8G2 = Vector256.LoadUnsafe(ref Unsafe.Add(ref q8Ref, q8Base + 64));
                    var q8G3 = Vector256.LoadUnsafe(ref Unsafe.Add(ref q8Ref, q8Base + 96));

                    unsignedTerm += GroupDot(qG0, q8G0, (sbyte)scales[scBase + 0], (sbyte)scales[scBase + 1], ones16)
                                  + GroupDot(qG1, q8G1, (sbyte)scales[scBase + 2], (sbyte)scales[scBase + 3], ones16)
                                  + GroupDot(qG2, q8G2, (sbyte)scales[scBase + 4], (sbyte)scales[scBase + 5], ones16)
                                  + GroupDot(qG3, q8G3, (sbyte)scales[scBase + 6], (sbyte)scales[scBase + 7], ones16);
                }
                return unsignedTerm;
            }

            // Scalar fallback — bit-identical (INT32 arithmetic, same order).
            var sum = 0;
            for (var h = 0; h < 2; h++)
            {
                var qlBase = 64 * h;
                var qhBase = 32 * h;
                var scBase = 8 * h;
                var q8Base = 128 * h;

                for (var l = 0; l < 32; l++)
                {
                    var qhByte = qh[qhBase + l];
                    var subBlockOffset = l >> 4;   // l / 16 ∈ {0, 1}

                    var q1 = (ql[qlBase + l] & 0x0F) | (((qhByte >> 0) & 0x03) << 4);
                    var q2 = (ql[qlBase + l + 32] & 0x0F) | (((qhByte >> 2) & 0x03) << 4);
                    var q3 = (ql[qlBase + l] >> 4) | (((qhByte >> 4) & 0x03) << 4);
                    var q4 = (ql[qlBase + l + 32] >> 4) | (((qhByte >> 6) & 0x03) << 4);

                    int s1 = (sbyte)scales[scBase + subBlockOffset];
                    int s2 = (sbyte)scales[scBase + subBlockOffset + 2];
                    int s3 = (sbyte)scales[scBase + subBlockOffset + 4];
                    int s4 = (sbyte)scales[scBase + subBlockOffset + 6];

                    sum += s1 * q1 * q8Block[q8Base + l]
                         + s2 * q2 * q8Block[q8Base + l + 32]
                         + s3 * q3 * q8Block[q8Base + l + 64]
                         + s4 * q4 * q8Block[q8Base + l + 96];
                }
            }
            return sum;
        }

        /// <summary>
        /// One AVX2 quant-group dot: 32 unsigned 6-bit quants × 32 q8 → two
        /// sub-block sums (16 elements each) multiplied by their respective
        /// signed int8 scales and added. <c>vpmaddubsw</c> pairs adjacent bytes
        /// (sum ≤ 2·63·128 = 16128, never saturates int16); <c>vpmaddwd</c>
        /// widens to int32. The lower 4 int32 sum sub-block A, upper 4 sum B.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GroupDot(
            Vector256<byte> q, Vector256<sbyte> q8,
            int scaleA, int scaleB, Vector256<short> ones)
        {
            var pairs16 = Avx2.MultiplyAddAdjacent(q, q8);
            var pairs32 = Avx2.MultiplyAddAdjacent(pairs16, ones);
            var sA = Vector128.Sum(pairs32.GetLower());
            var sB = Vector128.Sum(pairs32.GetUpper());
            return scaleA * sA + scaleB * sB;
        }

        /// <summary>
        /// Quantized single-token projection: <c>output = bias + input @ W</c>,
        /// W resident as Q6_K (output-major). The F32 <paramref name="input"/> is
        /// quantized once to Q8_K into the caller-owned scratch, then each output
        /// is the sum of <see cref="Dot"/> over its super-blocks. Sequential
        /// reference shape; the decode path uses <see cref="ProjectParallel"/>.
        /// </summary>
        public static void Project(
            ReadOnlySpan<float> input,
            Q6KWeight weight,
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
        /// shared input once and run several projections against it without
        /// re-quantizing (e.g. the attention <c>hidden</c> row reused across heads).
        /// </summary>
        public static void ProjectPreQuantized(
            Q6KWeight weight,
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
                var rowBase = (long)o * superBlocksPerRow * Q6KWeight.SuperBlockBytes;
                var sum = 0f;
                for (var sb = 0; sb < superBlocksPerRow; sb++)
                {
                    var block = blocks.Slice(
                        (int)(rowBase + (long)sb * Q6KWeight.SuperBlockBytes), Q6KWeight.SuperBlockBytes);
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
        /// loop split across the zero-allocation <c>OverfitParallel</c> worker
        /// pool. The activation is quantized once (sequentially) into the
        /// caller-owned scratch, then each worker computes a disjoint band of
        /// output dots. Bit-identical to <see cref="Project"/>.
        /// </summary>
        public static void ProjectParallel(
            ReadOnlySpan<float> input,
            Q6KWeight weight,
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
                var context = new Q6KProjectContext
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

                OverfitParallel.ForDecode(0, outputSize, &ProjectChunk, &context);
            }
        }

        /// <summary>Worker body for <see cref="ProjectParallel"/> — one disjoint band of output rows.</summary>
        private static void ProjectChunk(int chunkStart, int chunkEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<Q6KProjectContext>(context);
            var superBlocksPerRow = ctx.SuperBlocksPerRow;

            for (var o = chunkStart; o < chunkEnd; o++)
            {
                var rowBase = (long)o * superBlocksPerRow * Q6KWeight.SuperBlockBytes;
                var sum = 0f;
                for (var sb = 0; sb < superBlocksPerRow; sb++)
                {
                    var block = new ReadOnlySpan<byte>(
                        ctx.Blocks + rowBase + (long)sb * Q6KWeight.SuperBlockBytes,
                        Q6KWeight.SuperBlockBytes);
                    var q8 = new ReadOnlySpan<sbyte>(
                        ctx.ActivationQuants + sb * SuperBlockElements, SuperBlockElements);
                    var bsums = new ReadOnlySpan<short>(
                        ctx.ActivationBsums + sb * GroupsPerSuperBlock, GroupsPerSuperBlock);

                    sum += Dot(block, q8, ctx.ActivationScales[sb], bsums);
                }

                ctx.Output[o] = ctx.BiasLength == 0 ? sum : ctx.Bias[o] + sum;
            }
        }

        private struct Q6KProjectContext
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
        /// Batched Q6_K projection: <paramref name="rows"/> activation rows × one weight matrix — the
        /// prefill counterpart of <see cref="ProjectParallel"/>. Each row is quantized to Q8_K once;
        /// the output loop is split across <c>OverfitParallel</c> with the <b>rows loop innermost</b>,
        /// so each weight output row's super-blocks are read from DRAM once and reused (cache-hot) across
        /// all <paramref name="rows"/> dots — cutting weight byte-traffic ~<paramref name="rows"/>× vs
        /// N× single-token <see cref="Project"/>. Bit-identical to N× <see cref="Project"/>.
        /// </summary>
        public static void ProjectBatched(
            ReadOnlySpan<float> input,
            int rows,
            Q6KWeight weight,
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
                var context = new Q6KBatchedContext
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

                OverfitParallel.For(0, outputSize, &ProjectBatchedChunk, &context);
            }
        }

        /// <summary>Worker for <see cref="ProjectBatched"/> — a band of output cols, rows innermost.</summary>
        private static void ProjectBatchedChunk(int chunkStart, int chunkEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<Q6KBatchedContext>(context);
            var superBlocksPerRow = ctx.SuperBlocksPerRow;

            for (var o = chunkStart; o < chunkEnd; o++)
            {
                var rowBase = (long)o * superBlocksPerRow * Q6KWeight.SuperBlockBytes;
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
                            ctx.Blocks + rowBase + (long)sb * Q6KWeight.SuperBlockBytes,
                            Q6KWeight.SuperBlockBytes);
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

        private struct Q6KBatchedContext
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
