// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// A weight matrix resident as GGUF Q4_K — ~4.5 bits per weight,
    /// <b>output-major</b>: row <c>o</c> (one output) holds that output's
    /// <see cref="InputSize"/>-long contraction vector as
    /// <c>InputSize / 256</c> Q4_K super-blocks (144 bytes / 256 values each).
    ///
    /// The on-disk <c>block_q4_K</c> layout is kept verbatim — 2 B FP16 scale
    /// <c>d</c>, 2 B FP16 min-scale <c>dmin</c>, 12 B of packed 6-bit sub-block
    /// scales/mins, 128 B of 4-bit quants. <see cref="Q4KDotKernel"/> consumes
    /// it directly (no dequantization to F32).
    ///
    /// Long-lived (one per quantized model weight); a plain byte array, no pooling.
    /// </summary>
    public sealed class Q4KWeight
    {
        /// <summary>Elements per Q4_K super-block.</summary>
        public const int SuperBlockElements = 256;

        /// <summary>Bytes per Q4_K super-block: 2(d) + 2(dmin) + 12(scales/mins) + 128(quants).</summary>
        public const int SuperBlockBytes = 144;

        public Q4KWeight(byte[] blocks, int inputSize, int outputSize)
        {
            ArgumentNullException.ThrowIfNull(blocks);
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
                    $"blocks array ({blocks.Length} B) is smaller than " +
                    $"outputSize * superBlocksPerRow * {SuperBlockBytes} ({expected} B).",
                    nameof(blocks));
            }

            Blocks = blocks;
            InputSize = inputSize;
            OutputSize = outputSize;
        }

        /// <summary>Raw Q4_K super-block bytes, output-major: <c>OutputSize</c> rows
        /// of <see cref="SuperBlocksPerRow"/> × <see cref="SuperBlockBytes"/>.</summary>
        public byte[] Blocks { get; }

        /// <summary>Contraction-dimension length (a multiple of <see cref="SuperBlockElements"/>).</summary>
        public int InputSize { get; }

        /// <summary>Number of outputs (rows).</summary>
        public int OutputSize { get; }

        /// <summary>Q4_K super-blocks per output row.</summary>
        public int SuperBlocksPerRow => InputSize / SuperBlockElements;

        /// <summary>Resident size in bytes.</summary>
        public long ByteCount => Blocks.Length;
    }
}
