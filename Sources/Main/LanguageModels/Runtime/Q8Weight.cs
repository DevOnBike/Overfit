// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

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
    public sealed class Q8Weight : Autograd.IDequantRowSource
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
        public sbyte[] Quants
        {
            get;
        }

        /// <summary>Per-block scales, output-major: <c>[OutputSize * InputSize / BlockSize]</c>.</summary>
        public float[] Scales
        {
            get;
        }

        /// <summary>Contraction-dimension length (a multiple of <c>Q8DotKernel.BlockSize</c>).</summary>
        public int InputSize
        {
            get;
        }

        /// <summary>Number of outputs (rows).</summary>
        public int OutputSize
        {
            get;
        }

        /// <summary>Resident size in bytes — quants + scales.</summary>
        public long ByteCount => Quants.Length + (long)Scales.Length * sizeof(float);

        /// <summary>
        /// Dequantizes output row <paramref name="row"/> — one output's <see cref="InputSize"/>-long
        /// contraction vector — into <paramref name="dst"/> as F32: <c>dst[i] = scale[block] * quant[i]</c>.
        /// Zero-allocation; the per-token embedding-lookup primitive for a Q8-resident table.
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

            var blocksPerRow = InputSize / Q8DotKernel.BlockSize;
            var qBase = (long)row * InputSize;
            var sBase = (long)row * blocksPerRow;

            for (var b = 0; b < blocksPerRow; b++)
            {
                var scale = Scales[sBase + b];
                var off = b * Q8DotKernel.BlockSize;
                for (var i = 0; i < Q8DotKernel.BlockSize; i++)
                {
                    dst[off + i] = scale * Quants[qBase + off + i];
                }
            }
        }

        /// <summary>
        /// Quantizes a row-major F32 weight — <paramref name="rowCount"/> rows of
        /// <paramref name="rowLength"/> — into a Q8_0 weight where each row is one
        /// output's contraction vector. <paramref name="rowLength"/> must be a
        /// multiple of <see cref="Q8DotKernel.BlockSize"/>.
        /// </summary>
#pragma warning disable OVERFIT001 // load-time: one-shot F32->Q8 quantization of a weight matrix
        public static unsafe Q8Weight QuantizeRows(ReadOnlySpan<float> rowMajor, int rowCount, int rowLength)
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

            // Per-row quantization is independent (disjoint reads + writes) —
            // split the row range across the zero-allocation worker pool. This
            // is a model-load hot path: ~⅓ of GGUF load time before this change
            // (see docs/llamacpp-cpu-analysis.md §5). Bit-identical to the
            // sequential row loop — each row's output depends only on that row.
            fixed (float* src = rowMajor)
            fixed (sbyte* quantsPtr = quants)
            fixed (float* scalesPtr = scales)
            {
                var context = new QuantizeRowsContext
                {
                    Source = src,
                    Quants = quantsPtr,
                    Scales = scalesPtr,
                    RowLength = rowLength,
                    BlocksPerRow = blocksPerRow,
                };

                OverfitParallel.For(0, rowCount, &QuantizeRowChunk, &context);
            }

            return new Q8Weight(quants, scales, rowLength, rowCount);
        }
#pragma warning restore OVERFIT001

        /// <summary>
        /// Builds a Q8 weight straight from a Q5_0 tensor's raw bytes, one row at a time, without ever
        /// materialising the F32 table.
        ///
        /// <para><b>Why this exists — `XC-97`.</b> <c>GgufLlamaLoader.LoadEmbedding</c> keeps the embedding
        /// verbatim and mmap-backed only for Q4_K and Q6_K; every other type fell through to a full dequant
        /// into <c>[vocab x dModel]</c> F32. On Qwen2.5-0.5B, which is Q5_0, that was **519 MB measured for
        /// a 469 MB model file** — more managed heap than the whole model on disk.</para>
        ///
        /// <para><b>Why Q8 rather than a native Q5_0 weight type.</b> A fifth <see cref="DecodeWeight"/> case
        /// would touch every switch in the runtime. Q8 already is a first-class case with kernels, and the
        /// conversion is near-lossless in the direction that matters: <b>a Q5_0 block carries 32 distinct
        /// levels and a Q8 block carries 256</b>, so the target has eight times the resolution the source
        /// actually uses. The only error is the second rounding of an already-quantized value.</para>
        ///
        /// <para><b>Peak, not just steady state.</b> The rows are decoded from <paramref name="raw"/> — a
        /// zero-copy view over the mapping when the caller passes one — into a per-worker scratch of ONE row,
        /// so the high-water mark is the output arrays alone. Converting via the whole F32 table would have
        /// halved the steady state while leaving the load-time spike untouched, and minimising PEAK RAM at
        /// load is the constraint this repository states for itself.</para>
        /// </summary>
#pragma warning disable OVERFIT001 // load-time: one-shot Q5_0->Q8 conversion of a weight matrix
        public static unsafe Q8Weight QuantizeQ5_0Rows(ReadOnlySpan<byte> raw, int rowCount, int rowLength)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowLength);

            if (rowLength % GgmlDequant.Q5_0_BlockElements != 0)
            {
                throw new ArgumentException(
                    $"rowLength ({rowLength}) must be a multiple of {GgmlDequant.Q5_0_BlockElements}.",
                    nameof(rowLength));
            }

            if (rowLength % Q8DotKernel.BlockSize != 0)
            {
                throw new ArgumentException(
                    $"rowLength ({rowLength}) must be a multiple of {Q8DotKernel.BlockSize}.", nameof(rowLength));
            }

            var q5BlocksPerRow = rowLength / GgmlDequant.Q5_0_BlockElements;
            var rowBytes = q5BlocksPerRow * GgmlDequant.Q5_0_BlockBytes;
            var needed = (long)rowCount * rowBytes;

            if (raw.Length < needed)
            {
                throw new ArgumentException(
                    $"raw holds {raw.Length} bytes, expected at least {needed}.", nameof(raw));
            }

            var blocksPerRow = rowLength / Q8DotKernel.BlockSize;
            var quants = new sbyte[checked((int)((long)rowCount * rowLength))];
            var scales = new float[checked((int)((long)rowCount * blocksPerRow))];

            fixed (byte* rawPtr = raw)
            fixed (sbyte* quantsPtr = quants)
            fixed (float* scalesPtr = scales)
            {
                var context = new ConvertQ5RowsContext
                {
                    Raw = rawPtr,
                    Quants = quantsPtr,
                    Scales = scalesPtr,
                    RowLength = rowLength,
                    RowBytes = rowBytes,
                    Q5BlocksPerRow = q5BlocksPerRow,
                    BlocksPerRow = blocksPerRow,
                };

                OverfitParallel.For(0, rowCount, &ConvertQ5RowChunk, &context);
            }

            return new Q8Weight(quants, scales, rowLength, rowCount);
        }
#pragma warning restore OVERFIT001

        /// <summary>
        /// One disjoint band of rows for <see cref="QuantizeQ5_0Rows"/>. The F32 scratch is rented ONCE per
        /// band rather than per row: a row is only a few kilobytes, but a per-row rental would turn a load
        /// into millions of pool operations.
        /// </summary>
        private static unsafe void ConvertQ5RowChunk(int rowStart, int rowEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<ConvertQ5RowsContext>(context);

            using var scratch = new PooledBuffer<float>(ctx.RowLength, clearMemory: false);

            var row = scratch.Span;

            for (var r = rowStart; r < rowEnd; r++)
            {
                var rowBase = (long)r * ctx.RowBytes;

                for (var b = 0; b < ctx.Q5BlocksPerRow; b++)
                {
                    GgmlDequant.DecodeQ5_0Block(
                        new ReadOnlySpan<byte>(
                            ctx.Raw + rowBase + ((long)b * GgmlDequant.Q5_0_BlockBytes),
                            GgmlDequant.Q5_0_BlockBytes),
                        row.Slice(b * GgmlDequant.Q5_0_BlockElements, GgmlDequant.Q5_0_BlockElements));
                }

                Q8DotKernel.Quantize(
                    row,
                    new Span<sbyte>(ctx.Quants + ((long)r * ctx.RowLength), ctx.RowLength),
                    new Span<float>(ctx.Scales + ((long)r * ctx.BlocksPerRow), ctx.BlocksPerRow));
            }
        }

        private unsafe struct ConvertQ5RowsContext
        {
            public byte* Raw;
            public sbyte* Quants;
            public float* Scales;
            public int RowLength;
            public int RowBytes;
            public int Q5BlocksPerRow;
            public int BlocksPerRow;
        }

        /// <summary>Worker body for <see cref="QuantizeRows"/> — one disjoint band of rows.</summary>
        private static unsafe void QuantizeRowChunk(int rowStart, int rowEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<QuantizeRowsContext>(context);
            var rowLength = ctx.RowLength;
            var blocksPerRow = ctx.BlocksPerRow;

            for (var r = rowStart; r < rowEnd; r++)
            {
                Q8DotKernel.Quantize(
                    new ReadOnlySpan<float>(ctx.Source + (long)r * rowLength, rowLength),
                    new Span<sbyte>(ctx.Quants + (long)r * rowLength, rowLength),
                    new Span<float>(ctx.Scales + (long)r * blocksPerRow, blocksPerRow));
            }
        }

        private unsafe struct QuantizeRowsContext
        {
            public float* Source;
            public sbyte* Quants;
            public float* Scales;
            public int RowLength;
            public int BlocksPerRow;
        }
    }
}
