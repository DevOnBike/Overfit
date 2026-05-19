// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// A weight matrix resident as Q8_0 — 8-bit quantized, <b>output-major</b>:
    /// row <c>o</c> (one output) holds that output's <see cref="InputSize"/>-long
    /// contraction vector, quantized in <c>InputSize / Q8DotKernel.BlockSize</c>
    /// blocks of 32. Storage is ~8.5 bits per weight (one <see cref="sbyte"/>
    /// plus one F32 scale per 32) vs 32 for F32 — the byte reduction that makes
    /// the bandwidth-bound decode matmul faster and shrinks RAM.
    ///
    /// Long-lived (one per quantized model weight); plain arrays, no pooling.
    /// </summary>
    public sealed class Q8Weight
    {
        public Q8Weight(sbyte[] quants, float[] scales, int inputSize, int outputSize)
        {
            ArgumentNullException.ThrowIfNull(quants);
            ArgumentNullException.ThrowIfNull(scales);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputSize);

            if (inputSize % Q8DotKernel.BlockSize != 0)
            {
                throw new ArgumentException(
                    $"inputSize ({inputSize}) must be a multiple of {Q8DotKernel.BlockSize}.", nameof(inputSize));
            }
            if (quants.Length < (long)outputSize * inputSize)
            {
                throw new ArgumentException("Quants array is smaller than outputSize * inputSize.", nameof(quants));
            }
            if (scales.Length < (long)outputSize * (inputSize / Q8DotKernel.BlockSize))
            {
                throw new ArgumentException("Scales array is smaller than outputSize * blocksPerRow.", nameof(scales));
            }

            Quants = quants;
            Scales = scales;
            InputSize = inputSize;
            OutputSize = outputSize;
        }

        /// <summary>Q8 quants, output-major: <c>[OutputSize * InputSize]</c>.</summary>
        public sbyte[] Quants { get; }

        /// <summary>Per-block scales, output-major: <c>[OutputSize * InputSize / BlockSize]</c>.</summary>
        public float[] Scales { get; }

        /// <summary>Contraction-dimension length (a multiple of <c>Q8DotKernel.BlockSize</c>).</summary>
        public int InputSize { get; }

        /// <summary>Number of outputs (rows).</summary>
        public int OutputSize { get; }

        /// <summary>Resident size in bytes — quants + scales.</summary>
        public long ByteCount => Quants.Length + (long)Scales.Length * sizeof(float);

        /// <summary>
        /// Quantizes a row-major F32 weight — <paramref name="rowCount"/> rows of
        /// <paramref name="rowLength"/> — into a Q8_0 weight where each row is one
        /// output's contraction vector. <paramref name="rowLength"/> must be a
        /// multiple of <see cref="Q8DotKernel.BlockSize"/>.
        /// </summary>
        public static Q8Weight QuantizeRows(ReadOnlySpan<float> rowMajor, int rowCount, int rowLength)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowLength);

            if (rowLength % Q8DotKernel.BlockSize != 0)
            {
                throw new ArgumentException(
                    $"rowLength ({rowLength}) must be a multiple of {Q8DotKernel.BlockSize}.", nameof(rowLength));
            }

            var total = (long)rowCount * rowLength;
            if (rowMajor.Length < total)
            {
                throw new ArgumentException("rowMajor span is smaller than rowCount * rowLength.", nameof(rowMajor));
            }

            var blocksPerRow = rowLength / Q8DotKernel.BlockSize;
            var quants = new sbyte[checked((int)total)];
            var scales = new float[checked((int)((long)rowCount * blocksPerRow))];

            for (var r = 0; r < rowCount; r++)
            {
                Q8DotKernel.Quantize(
                    rowMajor.Slice(r * rowLength, rowLength),
                    quants.AsSpan(r * rowLength, rowLength),
                    scales.AsSpan(r * blocksPerRow, blocksPerRow));
            }

            return new Q8Weight(quants, scales, rowLength, rowCount);
        }
    }
}
