// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;

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
    public sealed class Q4KWeight : Autograd.IDequantRowSource
    {
        /// <summary>Elements per Q4_K super-block.</summary>
        public const int SuperBlockElements = 256;

        /// <summary>Bytes per Q4_K super-block: 2(d) + 2(dmin) + 12(scales/mins) + 128(quants).</summary>
        public const int SuperBlockBytes = 144;

        /// <summary>
        /// Backs the weight with a managed <c>byte[]</c> (the copy path — convert / per-head split).
        /// </summary>
        public Q4KWeight(byte[] blocks, int inputSize, int outputSize)
            : this((ReadOnlyMemory<byte>)(blocks ?? throw new ArgumentNullException(nameof(blocks))), inputSize, outputSize)
        {
        }

        /// <summary>
        /// Backs the weight with an arbitrary memory region — a managed array OR a slice of a
        /// memory-mapped GGUF file (zero-copy; Q4_K's on-disk layout is kept verbatim). The
        /// region's owner must outlive this weight.
        /// </summary>
        public Q4KWeight(ReadOnlyMemory<byte> blocks, int inputSize, int outputSize)
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

        /// <summary>Raw Q4_K super-block bytes, output-major: <c>OutputSize</c> rows
        /// of <see cref="SuperBlocksPerRow"/> × <see cref="SuperBlockBytes"/>. Backed by a managed
        /// array or a memory-mapped slice; read via <see cref="BlockSpan"/>.</summary>
        public ReadOnlyMemory<byte> Blocks
        {
            get;
        }

        /// <summary>The block bytes as a span (managed array or memory-mapped region).</summary>
        public ReadOnlySpan<byte> BlockSpan => Blocks.Span;

        /// <summary>Contraction-dimension length (a multiple of <see cref="SuperBlockElements"/>).</summary>
        public int InputSize
        {
            get;
        }

        /// <summary>Number of outputs (rows).</summary>
        public int OutputSize
        {
            get;
        }

        /// <summary>Q4_K super-blocks per output row.</summary>
        public int SuperBlocksPerRow => InputSize / SuperBlockElements;

        /// <summary>Resident size in bytes.</summary>
        public long ByteCount => Blocks.Length;

        private ReadOnlyMemory<byte>? _repacked;
        private bool _hasPrepacked;

        /// <summary>True when this weight can be repacked to <c>block_q4_Kx8</c> for the fast
        /// 8×8 decode GEMV (<see cref="Q4KGemvKernel"/>).</summary>
        public bool CanRepack => OutputSize % Q4KRepack.RowsInterleaved == 0 && InputSize % SuperBlockElements == 0;

        /// <summary>True when a pre-repacked view has been attached (see <see cref="SetPrepacked"/>) — i.e. the
        /// <c>block_q4_Kx8</c> layout is already resident (memory-mapped, zero heap), so routing through the
        /// repacked kernels costs no extra RAM and can be default-on.</summary>
        public bool IsPrepacked => _hasPrepacked;

        /// <summary>Bytes the <c>block_q4_Kx8</c> repacked form occupies for this shape.</summary>
        internal long RepackedByteCount =>
            (long)(OutputSize / Q4KRepack.RowsInterleaved) * SuperBlocksPerRow * Q4KRepack.BlockKx8Bytes;

        /// <summary>
        /// Supplies an already-<c>block_q4_Kx8</c>-repacked view (e.g. a zero-copy memory-mapped slice from an
        /// offline <see cref="Loading.RepackedWeightsFile"/> sidecar) so <see cref="EnsureRepacked"/> hands it
        /// out directly instead of building a heap copy — the RAM-free default-on path. The region must outlive
        /// this weight. Must be the exact repacked size and match <see cref="CanRepack"/>.
        /// </summary>
        internal void SetPrepacked(ReadOnlyMemory<byte> repacked)
        {
            if (!CanRepack)
            {
                throw new OverfitRuntimeException(
                    $"Q4KWeight {InputSize}x{OutputSize} is not repackable; cannot attach a pre-repacked view.");
            }
            if (repacked.Length != RepackedByteCount)
            {
                throw new OverfitFormatException(
                    $"Pre-repacked region is {repacked.Length} B, expected {RepackedByteCount} B for {InputSize}x{OutputSize}.");
            }
            _repacked = repacked;
            _hasPrepacked = true;
        }

        /// <summary>
        /// Lazily builds + caches the <c>block_q4_Kx8</c> repacked form (8 rows interleaved) for
        /// <see cref="Q4KGemvKernel.GemvParallel"/>. Returns a pre-attached view (see
        /// <see cref="SetPrepacked"/>) when present; otherwise allocates a copy (~same size as the original)
        /// on first call. Safe to call repeatedly. Requires <see cref="CanRepack"/>.
        /// </summary>
        public ReadOnlySpan<byte> EnsureRepacked()
        {
            if (_repacked is null)
            {
                ReadOnlyMemory<byte> built = Q4KRepack.RepackMatrix(BlockSpan, OutputSize, InputSize);
                _repacked = built;
            }
            return _repacked.Value.Span;
        }

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
                GgmlDequant.DecodeQ4_KBlock(block, dst.Slice(sb * SuperBlockElements, SuperBlockElements));
            }
        }

        /// <summary>True when the bytes are a memory-mapped region (no managed-heap copy).</summary>
        public bool IsMemoryMapped => System.Runtime.InteropServices.MemoryMarshal.TryGetArray(Blocks, out _) == false;
    }
}
