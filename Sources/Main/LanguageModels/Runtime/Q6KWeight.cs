// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// A weight matrix resident as GGUF Q6_K — ~6.5 bits per weight,
    /// <b>output-major</b>: row <c>o</c> (one output) holds that output's
    /// <see cref="InputSize"/>-long contraction vector as
    /// <c>InputSize / 256</c> Q6_K super-blocks (210 bytes / 256 values each).
    ///
    /// The on-disk <c>block_q6_K</c> layout is kept verbatim — 128 B of low
    /// 4-bit quants (<c>ql</c>), 64 B of high 2-bit quants (<c>qh</c>),
    /// 16 × int8 sub-block scales (<i>signed</i>, one per 16-element sub-block),
    /// then a 2 B FP16 super-scale <c>d</c>. <see cref="Q6KDotKernel"/> consumes
    /// it directly (no dequantization to F32).
    ///
    /// In Q4_K_M files Q6_K is used for the "boosted" tensors —
    /// <c>ffn_down</c>, <c>attn_v</c>, <c>token_embd</c>, <c>output</c> — where
    /// the higher precision matters most for quality.
    ///
    /// Long-lived (one per quantized model weight); a plain byte array, no pooling.
    /// </summary>
    public sealed class Q6KWeight
    {
        /// <summary>Elements per Q6_K super-block.</summary>
        public const int SuperBlockElements = 256;

        /// <summary>Bytes per Q6_K super-block: 128 ql + 64 qh + 16 scales + 2 d.</summary>
        public const int SuperBlockBytes = 210;

        /// <summary>Backs the weight with a managed <c>byte[]</c> (the copy path).</summary>
        public Q6KWeight(byte[] blocks, int inputSize, int outputSize)
            : this((ReadOnlyMemory<byte>)(blocks ?? throw new ArgumentNullException(nameof(blocks))), inputSize, outputSize)
        {
        }

        /// <summary>
        /// Backs the weight with an arbitrary memory region — a managed array OR a slice of a
        /// memory-mapped GGUF file (zero-copy; Q6_K's layout is kept verbatim). The region's
        /// owner must outlive this weight.
        /// </summary>
        public Q6KWeight(ReadOnlyMemory<byte> blocks, int inputSize, int outputSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputSize);

            if (inputSize % SuperBlockElements != 0)
            {
                throw new ArgumentException(
                    $"inputSize ({inputSize}) must be a multiple of {SuperBlockElements}.", nameof(inputSize));
            }

            var expected = (long)outputSize * (inputSize / SuperBlockElements) * SuperBlockBytes;
            if (blocks.Length < expected)
            {
                throw new ArgumentException(
                    $"blocks region ({blocks.Length} B) is smaller than " +
                    $"outputSize * superBlocksPerRow * {SuperBlockBytes} ({expected} B).",
                    nameof(blocks));
            }

            Blocks = blocks;
            InputSize = inputSize;
            OutputSize = outputSize;
        }

        /// <summary>Raw Q6_K super-block bytes, output-major; read via <see cref="BlockSpan"/>.</summary>
        public ReadOnlyMemory<byte> Blocks { get; }

        /// <summary>The block bytes as a span (managed array or memory-mapped region).</summary>
        public ReadOnlySpan<byte> BlockSpan => Blocks.Span;

        /// <summary>Contraction-dimension length (a multiple of <see cref="SuperBlockElements"/>).</summary>
        public int InputSize { get; }

        /// <summary>Number of outputs (rows).</summary>
        public int OutputSize { get; }

        /// <summary>Q6_K super-blocks per output row.</summary>
        public int SuperBlocksPerRow => InputSize / SuperBlockElements;

        /// <summary>Resident size in bytes.</summary>
        public long ByteCount => Blocks.Length;

        /// <summary>
        /// Dequantizes output row <paramref name="row"/> — one output's <see cref="InputSize"/>-long
        /// contraction vector, i.e. one token's embedding when this matrix is a token-embedding table —
        /// into <paramref name="dst"/> as F32. Decodes only that row's super-blocks straight from
        /// <see cref="BlockSpan"/> (no full-tensor dequant, no allocation), so it is cheap enough for
        /// the per-token lookup hot path.
        /// </summary>
        public void DecodeRow(int row, Span<float> dst)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(row);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, OutputSize);
            if (dst.Length < InputSize)
            {
                throw new ArgumentException(
                    $"Destination span too small: {dst.Length} < {InputSize}.", nameof(dst));
            }

            var blocksPerRow = SuperBlocksPerRow;
            var blocks = BlockSpan;
            var rowBase = (long)row * blocksPerRow * SuperBlockBytes;
            for (var sb = 0; sb < blocksPerRow; sb++)
            {
                var block = blocks.Slice((int)(rowBase + (long)sb * SuperBlockBytes), SuperBlockBytes);
                GgmlDequant.DecodeQ6_KBlock(block, dst.Slice(sb * SuperBlockElements, SuperBlockElements));
            }
        }
    }
}
