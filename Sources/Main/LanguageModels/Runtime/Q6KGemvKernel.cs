// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Repacked 8×8 Q6_K GEMV — 8 output rows from a <see cref="Q6KRepack"/> <c>block_q6_Kx8</c>
    /// weight and a Q8_K activation. llama.cpp has NO x86 AVX2 Q6_K 8×8 kernel (only ARM NEON +
    /// a generic scalar), so this is a NOVEL kernel: <see cref="GemvScalar"/> is the faithful port
    /// of the generic (the correctness oracle); the AVX2 path is built against it.
    /// </summary>
    public static class Q6KGemvKernel
    {
        private const int BlockKx8Bytes = Q6KRepack.BlockKx8Bytes; // 1680
        private const int DstScalesOffset = 16;
        private const int DstQlOffset = 144;
        private const int DstQhOffset = 1168;

        /// <summary>
        /// Scalar reference GEMV over the repacked layout — port of llama.cpp
        /// <c>ggml_gemv_q6_K_NxM_q8_K_generic_impl&lt;8,8&gt;</c>. Q6_K applies the <c>−32</c>
        /// bias per element (no bsums term). The correctness oracle for the AVX2 path.
        /// </summary>
        public static void GemvScalar(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            Span<float> output)
        {
            var nb = inputSize / 256;
            Span<float> sumf = stackalloc float[8];

            for (var x = 0; x < outputSize / 8; x++)
            {
                var bptr = (long)x * nb * BlockKx8Bytes;
                sumf.Clear();

                for (var l = 0; l < nb; l++)
                {
                    var blk = repacked.Slice((int)(bptr + (long)l * BlockKx8Bytes), BlockKx8Bytes);
                    var scales = blk.Slice(DstScalesOffset, 128);
                    var ql = blk.Slice(DstQlOffset, 1024);
                    var qh = blk.Slice(DstQhOffset, 512);

                    var aD = actScales[l];
                    var aqs = actQuants.Slice(l * 256, 256);

                    for (var k = 0; k < 16; k++)
                    {
                        var baseL = (k / 8) * 128 + (k % 8) * 8;
                        var baseH = baseL + 64;
                        var scaleIdxL = baseL / 16;
                        var scaleIdxH = baseH / 16;
                        var qhShiftL = ((baseL % 128) / 32) * 2;
                        var qhShiftH = ((baseH % 128) / 32) * 2;
                        var qhHalfL = (baseL / 128) * 32;
                        var qhHalfH = (baseH / 128) * 32;

                        for (var j = 0; j < 8; j++)
                        {
                            int scaleL = (sbyte)scales[scaleIdxL * 8 + j];
                            int scaleH = (sbyte)scales[scaleIdxH * 8 + j];

                            var sumiL = 0;
                            var sumiH = 0;

                            for (var i = 0; i < 8; i++)
                            {
                                var qlPos = k * 64 + j * 8 + i;
                                var l4 = ql[qlPos] & 0xF;
                                var hi4 = (ql[qlPos] >> 4) & 0xF;

                                var qhIdxL = qhHalfL + ((baseL + i) % 32);
                                var qhOffL = (qhIdxL / 8) * 64 + j * 8 + (qhIdxL % 8);
                                var hi2L = (qh[qhOffL] >> qhShiftL) & 0x3;

                                var qhIdxH = qhHalfH + ((baseH + i) % 32);
                                var qhOffH = (qhIdxH / 8) * 64 + j * 8 + (qhIdxH % 8);
                                var hi2H = (qh[qhOffH] >> qhShiftH) & 0x3;

                                var qL = ((hi2L << 4) | l4) - 32;
                                var qH = ((hi2H << 4) | hi4) - 32;

                                sumiL += qL * aqs[baseL + i];
                                sumiH += qH * aqs[baseH + i];
                            }

                            var d = (float)BitConverter.UInt16BitsToHalf(
                                MemoryMarshal.Read<ushort>(blk.Slice(j * 2, 2)));
                            sumf[j] += (sumiL * scaleL + sumiH * scaleH) * d * aD;
                        }
                    }
                }

                for (var j = 0; j < 8; j++)
                {
                    output[x * 8 + j] = sumf[j];
                }
            }
        }

        /// <summary>
        /// AVX2 8×8 Q6_K GEMV — vectorised over the 8 output rows (lanes), no per-row
        /// horizontal reduction. Same math as <see cref="GemvScalar"/>: forms each unsigned
        /// 6-bit weight (ql nibble | qh 2-bit &lt;&lt; 4), accumulates Σ q·a per row, applies the
        /// −32 bias via Σa, scales (signed int8), and the per-super-block d·activationScale.
        /// Validated bit-close to <see cref="GemvScalar"/>.
        /// </summary>
        /// <summary>
        /// Opt-in flag (shared with <see cref="Q4KGemvKernel.Enabled"/> via
        /// <c>OVERFIT_REPACK_GEMV</c>) for the repacked 8×8 decode GEMV. AVX2-only.
        /// </summary>
        public static bool Enabled => Q4KGemvKernel.Enabled;

        public static unsafe void GemvAvx2(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            Span<float> output)
        {
            var nb = inputSize / 256;
            fixed (byte* rep = repacked)
            fixed (sbyte* aq = actQuants)
            fixed (float* asc = actScales)
            fixed (float* o = output)
            {
                ComputeGroupRange(rep, aq, asc, o, 0, outputSize / 8, nb);
            }
        }

        /// <summary>
        /// Parallel full-matrix GEMV — output-row-groups split across the decode dispatch
        /// (capped / spin-pool). The <c>block_q6_Kx8</c> layout is row-group-major, so each
        /// worker owns a disjoint contiguous slice of weights + outputs.
        /// </summary>
        public static unsafe void GemvParallel(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            Span<float> output)
        {
            var nb = inputSize / 256;
            var groups = outputSize / 8;
            fixed (byte* rep = repacked)
            fixed (sbyte* aq = actQuants)
            fixed (float* asc = actScales)
            fixed (float* o = output)
            {
                var ctx = new GemvContext { Rep = rep, Aq = aq, Asc = asc, Out = o, Nb = nb };
                OverfitParallel.ForDecode(0, groups, &GroupChunk, &ctx);
            }
        }

        private unsafe struct GemvContext
        {
            public byte* Rep;
            public sbyte* Aq;
            public float* Asc;
            public float* Out;
            public int Nb;
        }

        private static unsafe void GroupChunk(int start, int end, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<GemvContext>(context);
            ComputeGroupRange(ctx.Rep, ctx.Aq, ctx.Asc, ctx.Out, start, end, ctx.Nb);
        }

        /// <summary>Output-row-groups [<paramref name="start"/>, <paramref name="end"/>) — the maddubs core.</summary>
        private static unsafe void ComputeGroupRange(
            byte* rep, sbyte* aqAll, float* asc, float* outp, int start, int end, int nb)
        {
            var m4b = Vector256.Create((byte)0x0F);
            var m2 = Vector256.Create((byte)0x03);
            var m32 = Vector256.Create((byte)32);
            var ones = Vector256.Create((short)1);
            var reduce = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);

            {
                for (var x = start; x < end; x++)
                {
                    var bptr = rep + (long)x * nb * BlockKx8Bytes;
                    var sumf = Vector256<float>.Zero;

                    for (var l = 0; l < nb; l++)
                    {
                        var blk = bptr + (long)l * BlockKx8Bytes;
                        var scales = blk + DstScalesOffset;
                        var ql = blk + DstQlOffset;
                        var qh = blk + DstQhOffset;
                        var dVec = LoadF16x8Int(blk);
                        var aD = asc[l];
                        var aqs = aqAll + l * 256;

                        var iacc = Vector256<int>.Zero;

                        for (var k = 0; k < 16; k++)
                        {
                            var baseL = (k / 8) * 128 + (k % 8) * 8;
                            var baseH = baseL + 64;
                            var qhShiftL = (byte)(((baseL % 128) / 32) * 2);
                            var qhShiftH = (byte)(((baseH % 128) / 32) * 2);
                            var qhHalfL = (baseL / 128) * 32;
                            var qhHalfH = (baseH / 128) * 32;
                            var qhBlockL = ((qhHalfL + (baseL % 32)) / 8) * 64;
                            var qhBlockH = ((qhHalfH + (baseH % 32)) / 8) * 64;

                            // Row-major loads: rows 0-3 in *03, rows 4-7 in *47 (8 bytes/row).
                            var ql03 = Vector256.Load(ql + k * 64);
                            var ql47 = Vector256.Load(ql + k * 64 + 32);
                            var qhL03 = Vector256.Load(qh + qhBlockL);
                            var qhL47 = Vector256.Load(qh + qhBlockL + 32);
                            var qhH03 = Vector256.Load(qh + qhBlockH);
                            var qhH47 = Vector256.Load(qh + qhBlockH + 32);

                            // L stream = low nibble | (qhL 2-bit << 4); H stream = high nibble | (qhH 2-bit << 4).
                            var qLu03 = Avx2.Or(LoNib(ql03, m4b), QhBits(qhL03, qhShiftL, m2));
                            var qLu47 = Avx2.Or(LoNib(ql47, m4b), QhBits(qhL47, qhShiftL, m2));
                            var qHu03 = Avx2.Or(HiNib(ql03, m4b), QhBits(qhH03, qhShiftH, m2));
                            var qHu47 = Avx2.Or(HiNib(ql47, m4b), QhBits(qhH47, qhShiftH, m2));

                            // Activations tiled so both rows in each 128-lane see a[base..base+8].
                            var actL = TileAct(aqs + baseL);
                            var actH = TileAct(aqs + baseH);

                            // maddubs (32 elems/op), fold −32 via maddubs(32, act), reduce to 8 row-lanes.
                            var sumL = ReduceRows(
                                Avx2.Subtract(Avx2.MultiplyAddAdjacent(qLu03, actL), Avx2.MultiplyAddAdjacent(m32, actL)),
                                Avx2.Subtract(Avx2.MultiplyAddAdjacent(qLu47, actL), Avx2.MultiplyAddAdjacent(m32, actL)),
                                ones, reduce);
                            var sumH = ReduceRows(
                                Avx2.Subtract(Avx2.MultiplyAddAdjacent(qHu03, actH), Avx2.MultiplyAddAdjacent(m32, actH)),
                                Avx2.Subtract(Avx2.MultiplyAddAdjacent(qHu47, actH), Avx2.MultiplyAddAdjacent(m32, actH)),
                                ones, reduce);

                            var scaleL = ScaleVec(scales + (baseL / 16) * 8);
                            var scaleH = ScaleVec(scales + (baseH / 16) * 8);

                            iacc = Avx2.Add(iacc, Avx2.Add(
                                Avx2.MultiplyLow(sumL, scaleL), Avx2.MultiplyLow(sumH, scaleH)));
                        }

                        sumf = Fma.MultiplyAdd(
                            Avx.ConvertToVector256Single(iacc), Avx.Multiply(dVec, Vector256.Create(aD)), sumf);
                    }

                    sumf.Store(outp + x * 8);
                }
            }
        }

        /// <summary>Max activation columns per <see cref="GemmTiled"/> call — the register-tile width. The
        /// caller splits a longer prompt into tiles of this many columns.</summary>
        public const int MaxTileCols = 16;

        /// <summary>
        /// Register-tiled Q6_K prefill GEMM over the repacked <c>block_q6_Kx8</c> layout: produces
        /// <paramref name="cols"/> output columns (prompt tokens) at once, unpacking each weight super-block
        /// <b>once</b> and reusing it across every column — the loop the decode <see cref="GemvAvx2"/> has
        /// nothing to tile.
        ///
        /// <para><b>Why this and not the weight-stationary shape.</b> A weight-stationary Q6_K kernel was
        /// built first, modelled on the Q4_K one, and measured <b>13.5% slower</b>: it hoisted the whole 6-bit
        /// unpack into a stack buffer, so each row paid a store+reload through L1 instead of consuming the
        /// quants from registers, and inverting the loops made activation reads strided. Tiling keeps the
        /// unpacked quants <i>in registers</i> and amortises them across columns instead — which is exactly
        /// why the Q4_K tiled kernel measures ~3.3× over its own weight-stationary variant.</para>
        ///
        /// <para><b>Bit-identical to <see cref="GemvAvx2"/> per column:</b> the per-(row, column) operation
        /// sequence and accumulation order are unchanged; only weight decoding moves outward. Layout matches
        /// the Q4_K tiled kernel — activations column-contiguous, output column-major
        /// (<c>output[c*outputSize + row]</c>). AVX2 + FMA.</para>
        /// </summary>
        /// <summary>Floats written per weight block by <see cref="DecodeBlockScales"/>: the eight row scales.</summary>
        public const int DecodedScalesPerBlock = 8;

        /// <summary>
        /// Widens every weight block's eight F16 row scales to <see cref="float"/> once, into
        /// <paramref name="destination"/> laid out as <c>[(group·nb + block) · 8]</c>.
        ///
        /// <para>The Q6_K counterpart of <c>Q4KGemvKernel.DecodeBlockScales</c>, and needed for the same
        /// reason: <see cref="GemmTiled"/> widens these inline once per (group, block), but the kernel itself
        /// runs once per <i>column tile</i>, so the same values are decoded as many times as there are tiles.
        /// On the Q4_K side hoisting this measured +13% on the projection. Q6_K carries `ffn_down`, 27% of
        /// prefill. Bit-identical: the same conversions, fewer times.</para>
        ///
        /// <para>Q6_K has no <c>dmin</c>, so this writes 8 floats per block against Q4_K's 16.</para>
        /// </summary>
        public static unsafe void DecodeBlockScales(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            Span<float> destination)
        {
            var nb = inputSize / 256;
            var groups = outputSize / 8;

            fixed (byte* w = repacked)
            fixed (float* d = destination)
            {
                for (var x = 0; x < groups; x++)
                {
                    for (var l = 0; l < nb; l++)
                    {
                        var index = ((long)x * nb + l) * DecodedScalesPerBlock;

                        LoadF16x8Int(w + ((long)x * nb + l) * BlockKx8Bytes).Store(d + index);
                    }
                }
            }
        }

        public static unsafe void GemmTiled(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            int cols,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            Span<float> output,
            ReadOnlySpan<float> decodedScales = default)
        {
            if (cols is < 1 or > MaxTileCols)
            {
                throw new ArgumentOutOfRangeException(nameof(cols), cols, $"cols must be in [1, {MaxTileCols}].");
            }

            var nb = inputSize / 256;

            var m4b = Vector256.Create((byte)0x0F);
            var m2 = Vector256.Create((byte)0x03);
            var m32 = Vector256.Create((byte)32);
            var ones = Vector256.Create((short)1);
            var reduce = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);

            // Per-column accumulators. cols <= MaxTileCols keeps this a small bounded frame.
            Span<Vector256<float>> sumf = stackalloc Vector256<float>[cols];
            Span<Vector256<int>> iacc = stackalloc Vector256<int>[cols];

            fixed (byte* rep = repacked)
            fixed (sbyte* aqAll = actQuants)
            fixed (float* asc = actScales)
            fixed (float* outp = output)
            fixed (float* dsc = decodedScales) // null when the caller did not pre-decode; see DecodeBlockScales
            {
                for (var x = 0; x < outputSize / 8; x++)
                {
                    var bptr = rep + (long)x * nb * BlockKx8Bytes;

                    for (var c = 0; c < cols; c++)
                    {
                        sumf[c] = Vector256<float>.Zero;
                    }

                    for (var l = 0; l < nb; l++)
                    {
                        var blk = bptr + (long)l * BlockKx8Bytes;
                        var scales = blk + DstScalesOffset;
                        var ql = blk + DstQlOffset;
                        var qh = blk + DstQhOffset;

                        // Pre-decoded when the caller hoisted the F16 widening out of the tile loop; identical
                        // values either way, so the two paths are bit-identical.
                        var dVec = dsc is not null
                            ? Vector256.Load(dsc + (((long)x * nb) + l) * DecodedScalesPerBlock)
                            : LoadF16x8Int(blk);

                        for (var c = 0; c < cols; c++)
                        {
                            iacc[c] = Vector256<int>.Zero;
                        }

                        for (var k = 0; k < 16; k++)
                        {
                            var baseL = (k / 8) * 128 + (k % 8) * 8;
                            var baseH = baseL + 64;
                            var qhShiftL = (byte)(((baseL % 128) / 32) * 2);
                            var qhShiftH = (byte)(((baseH % 128) / 32) * 2);
                            var qhHalfL = (baseL / 128) * 32;
                            var qhHalfH = (baseH / 128) * 32;
                            var qhBlockL = ((qhHalfL + (baseL % 32)) / 8) * 64;
                            var qhBlockH = ((qhHalfH + (baseH % 32)) / 8) * 64;

                            // ── Weight side: decoded ONCE, reused across every column (the tiling win). ──
                            var ql03 = Vector256.Load(ql + k * 64);
                            var ql47 = Vector256.Load(ql + k * 64 + 32);
                            var qhL03 = Vector256.Load(qh + qhBlockL);
                            var qhL47 = Vector256.Load(qh + qhBlockL + 32);
                            var qhH03 = Vector256.Load(qh + qhBlockH);
                            var qhH47 = Vector256.Load(qh + qhBlockH + 32);

                            var qLu03 = Avx2.Or(LoNib(ql03, m4b), QhBits(qhL03, qhShiftL, m2));
                            var qLu47 = Avx2.Or(LoNib(ql47, m4b), QhBits(qhL47, qhShiftL, m2));
                            var qHu03 = Avx2.Or(HiNib(ql03, m4b), QhBits(qhH03, qhShiftH, m2));
                            var qHu47 = Avx2.Or(HiNib(ql47, m4b), QhBits(qhH47, qhShiftH, m2));

                            var scaleL = ScaleVec(scales + (baseL / 16) * 8);
                            var scaleH = ScaleVec(scales + (baseH / 16) * 8);

                            // ── Activation side: per column. ──
                            for (var c = 0; c < cols; c++)
                            {
                                var aqs = aqAll + (long)c * inputSize + l * 256;
                                var actL = TileAct(aqs + baseL);
                                var actH = TileAct(aqs + baseH);

                                var sumL = ReduceRows(
                                    Avx2.Subtract(Avx2.MultiplyAddAdjacent(qLu03, actL), Avx2.MultiplyAddAdjacent(m32, actL)),
                                    Avx2.Subtract(Avx2.MultiplyAddAdjacent(qLu47, actL), Avx2.MultiplyAddAdjacent(m32, actL)),
                                    ones, reduce);
                                var sumH = ReduceRows(
                                    Avx2.Subtract(Avx2.MultiplyAddAdjacent(qHu03, actH), Avx2.MultiplyAddAdjacent(m32, actH)),
                                    Avx2.Subtract(Avx2.MultiplyAddAdjacent(qHu47, actH), Avx2.MultiplyAddAdjacent(m32, actH)),
                                    ones, reduce);

                                iacc[c] = Avx2.Add(iacc[c], Avx2.Add(
                                    Avx2.MultiplyLow(sumL, scaleL), Avx2.MultiplyLow(sumH, scaleH)));
                            }
                        }

                        for (var c = 0; c < cols; c++)
                        {
                            sumf[c] = Fma.MultiplyAdd(
                                Avx.ConvertToVector256Single(iacc[c]),
                                Avx.Multiply(dVec, Vector256.Create(asc[(long)c * nb + l])),
                                sumf[c]);
                        }
                    }

                    for (var c = 0; c < cols; c++)
                    {
                        sumf[c].Store(outp + (long)c * outputSize + x * 8);
                    }
                }
            }
        }

        /// <summary>
        /// AVX-512 form of <see cref="GemmTiled"/>, structured exactly like
        /// <c>Q4KGemvKernel.GemmTiled512</c>: <b>two activation columns per instruction</b>, column <c>2p</c>
        /// in the low 256 bits and <c>2p+1</c> in the high, with the shared weights broadcast into both halves
        /// and the pair loop left innermost so the per-block weight decode stays amortised across the tile.
        ///
        /// <para><b>The one place it cannot widen.</b> <see cref="ReduceRows"/> ends in <c>vphaddd</c>, which
        /// AVX-512 does not provide for zmm at all. The two <c>vpmaddwd</c> steps still run at 512; only the
        /// horizontal add and its permute drop to 256-bit halves and are re-joined. That costs one extra
        /// instruction per reduction against eighteen saved elsewhere in the same iteration — the arithmetic
        /// bulk (eight <c>vpmaddubsw</c>, four subtracts, the scale multiplies) all pairs cleanly.</para>
        ///
        /// <para><b>Bit-identical</b> to <see cref="GemmTiled"/>: each column's operations and their order are
        /// unchanged, including the split reduction, which is the same 256-bit sequence applied to the same
        /// values.</para>
        /// </summary>
        public static unsafe void GemmTiled512(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            int cols,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            Span<float> output,
            ReadOnlySpan<float> decodedScales = default)
        {
            if (cols is < 1 or > MaxTileCols)
            {
                throw new ArgumentOutOfRangeException(nameof(cols), cols, $"cols must be in [1, {MaxTileCols}].");
            }

            var nb = inputSize / 256;
            var pairs = (cols + 1) / 2;

            var m4b = Vector512.Create((byte)0x0F);
            var m2 = Vector512.Create((byte)0x03);
            var m32 = Vector512.Create((byte)32);
            var ones = Vector512.Create((short)1);
            var reduce = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);

            Span<Vector512<float>> sumf = stackalloc Vector512<float>[pairs];
            Span<Vector512<int>> iacc = stackalloc Vector512<int>[pairs];

            fixed (byte* rep = repacked)
            fixed (sbyte* aqAll = actQuants)
            fixed (float* asc = actScales)
            fixed (float* outp = output)
            fixed (float* dsc = decodedScales)
            {
                for (var x = 0; x < outputSize / 8; x++)
                {
                    var bptr = rep + (long)x * nb * BlockKx8Bytes;

                    for (var p = 0; p < pairs; p++)
                    {
                        sumf[p] = Vector512<float>.Zero;
                    }

                    for (var l = 0; l < nb; l++)
                    {
                        var blk = bptr + (long)l * BlockKx8Bytes;
                        var scales = blk + DstScalesOffset;
                        var ql = blk + DstQlOffset;
                        var qh = blk + DstQhOffset;

                        var d256 = dsc is not null
                            ? Vector256.Load(dsc + (((long)x * nb) + l) * DecodedScalesPerBlock)
                            : LoadF16x8Int(blk);
                        var dVec = Vector512.Create(d256, d256);

                        for (var p = 0; p < pairs; p++)
                        {
                            iacc[p] = Vector512<int>.Zero;
                        }

                        for (var k = 0; k < 16; k++)
                        {
                            var baseL = (k / 8) * 128 + (k % 8) * 8;
                            var baseH = baseL + 64;
                            var qhShiftL = (byte)(((baseL % 128) / 32) * 2);
                            var qhShiftH = (byte)(((baseH % 128) / 32) * 2);
                            var qhHalfL = (baseL / 128) * 32;
                            var qhHalfH = (baseH / 128) * 32;
                            var qhBlockL = ((qhHalfL + (baseL % 32)) / 8) * 64;
                            var qhBlockH = ((qhHalfH + (baseH % 32)) / 8) * 64;

                            // Weight side: decoded once, shared by both columns of every pair.
                            var ql03 = Broadcast512(ql + k * 64);
                            var ql47 = Broadcast512(ql + k * 64 + 32);
                            var qhL03 = Broadcast512(qh + qhBlockL);
                            var qhL47 = Broadcast512(qh + qhBlockL + 32);
                            var qhH03 = Broadcast512(qh + qhBlockH);
                            var qhH47 = Broadcast512(qh + qhBlockH + 32);

                            var qLu03 = LoNib512(ql03, m4b) | QhBits512(qhL03, qhShiftL, m2);
                            var qLu47 = LoNib512(ql47, m4b) | QhBits512(qhL47, qhShiftL, m2);
                            var qHu03 = HiNib512(ql03, m4b) | QhBits512(qhH03, qhShiftH, m2);
                            var qHu47 = HiNib512(ql47, m4b) | QhBits512(qhH47, qhShiftH, m2);

                            var sl = ScaleVec(scales + (baseL / 16) * 8);
                            var sh = ScaleVec(scales + (baseH / 16) * 8);
                            var scaleL = Vector512.Create(sl, sl);
                            var scaleH = Vector512.Create(sh, sh);

                            for (var p = 0; p < pairs; p++)
                            {
                                var lowCol = 2 * p;
                                var highCol = Math.Min(2 * p + 1, cols - 1);
                                var aLow = aqAll + (long)lowCol * inputSize + l * 256;
                                var aHigh = aqAll + (long)highCol * inputSize + l * 256;

                                var actL = TileAct512(aLow + baseL, aHigh + baseL);
                                var actH = TileAct512(aLow + baseH, aHigh + baseH);

                                var sumL = ReduceRows512(
                                    Avx512BW.Subtract(Mul512(qLu03, actL), Mul512(m32, actL)),
                                    Avx512BW.Subtract(Mul512(qLu47, actL), Mul512(m32, actL)),
                                    ones, reduce);
                                var sumH = ReduceRows512(
                                    Avx512BW.Subtract(Mul512(qHu03, actH), Mul512(m32, actH)),
                                    Avx512BW.Subtract(Mul512(qHu47, actH), Mul512(m32, actH)),
                                    ones, reduce);

                                iacc[p] = Avx512F.Add(iacc[p], Avx512F.Add(
                                    Avx512F.MultiplyLow(sumL, scaleL), Avx512F.MultiplyLow(sumH, scaleH)));
                            }
                        }

                        for (var p = 0; p < pairs; p++)
                        {
                            var rowScale = Vector512.Create(
                                Vector256.Create(asc[(long)(2 * p) * nb + l]),
                                Vector256.Create(asc[(long)Math.Min(2 * p + 1, cols - 1) * nb + l]));

                            sumf[p] = Avx512F.FusedMultiplyAdd(
                                Avx512F.ConvertToVector512Single(iacc[p]),
                                Avx512F.Multiply(dVec, rowScale),
                                sumf[p]);
                        }
                    }

                    for (var p = 0; p < pairs; p++)
                    {
                        sumf[p].GetLower().Store(outp + (long)(2 * p) * outputSize + x * 8);

                        // The odd tail duplicated its low column into the high half; discard that copy.
                        if (2 * p + 1 < cols)
                        {
                            sumf[p].GetUpper().Store(outp + (long)(2 * p + 1) * outputSize + x * 8);
                        }
                    }
                }
            }
        }

        /// <summary>The same 32 weight bytes in both halves — weights are shared by the two columns of a pair.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<byte> Broadcast512(byte* p)
        {
            var v = Vector256.Load(p);

            return Vector512.Create(v, v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<byte> LoNib512(Vector512<byte> v, Vector512<byte> m4b) => v & m4b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<byte> HiNib512(Vector512<byte> v, Vector512<byte> m4b) =>
            Avx512BW.ShiftRightLogical(v.AsInt16(), 4).AsByte() & m4b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<byte> QhBits512(Vector512<byte> qh, byte shift, Vector512<byte> m2)
        {
            var count = Vector128.CreateScalar((short)shift);
            var bits = Avx512BW.ShiftRightLogical(qh.AsInt16(), count).AsByte() & m2;

            return Avx512BW.ShiftLeftLogical(bits.AsInt16(), 4).AsByte();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<short> Mul512(Vector512<byte> rhs, Vector512<sbyte> lhs) =>
            Avx512BW.MultiplyAddAdjacent(rhs, lhs);

        /// <summary>One column's eight activation bytes per half, duplicated in both 128-bit lanes of that half.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<sbyte> TileAct512(sbyte* low, sbyte* high)
        {
            var lo = Unsafe.ReadUnaligned<long>(low);
            var hi = Unsafe.ReadUnaligned<long>(high);
            var vl = Vector128.Create(lo, lo).AsSByte();
            var vh = Vector128.Create(hi, hi).AsSByte();

            return Vector512.Create(Vector256.Create(vl, vl), Vector256.Create(vh, vh));
        }

        /// <summary>
        /// <see cref="ReduceRows"/> widened as far as the instruction set allows: both <c>vpmaddwd</c> steps run
        /// at 512 bits, then the horizontal add and its permute drop to 256-bit halves because AVX-512 has no
        /// <c>vphaddd</c> for zmm. Each half is the identical 256-bit sequence, so the result is bit-identical.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<int> ReduceRows512(
            Vector512<short> p03, Vector512<short> p47, Vector512<short> ones, Vector256<int> reduce)
        {
            var m03 = Avx512BW.MultiplyAddAdjacent(p03, ones);
            var m47 = Avx512BW.MultiplyAddAdjacent(p47, ones);

            var lower = Avx2.PermuteVar8x32(Avx2.HorizontalAdd(m03.GetLower(), m47.GetLower()), reduce);
            var upper = Avx2.PermuteVar8x32(Avx2.HorizontalAdd(m03.GetUpper(), m47.GetUpper()), reduce);

            return Vector512.Create(lower, upper);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> LoNib(Vector256<byte> v, Vector256<byte> m4b) => Avx2.And(v, m4b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> HiNib(Vector256<byte> v, Vector256<byte> m4b) =>
            Avx2.And(Avx2.ShiftRightLogical(v.AsInt16(), 4).AsByte(), m4b);

        // 2-bit qh field at `shift`, masked, shifted into the high-nibble position (<<4).
        // `shift` is runtime-variable (0/2/4/6 per sub-block) → use the Vector128-count shift
        // overload (the variable-count form; the immediate-byte overload triggers CA1857).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> QhBits(Vector256<byte> qh, byte shift, Vector256<byte> m2)
        {
            var count = Vector128.CreateScalar((short)shift);
            var bits = Avx2.And(Avx2.ShiftRightLogical(qh.AsInt16(), count).AsByte(), m2);

            return Avx2.ShiftLeftLogical(bits.AsInt16(), 4).AsByte();
        }

        // Broadcast 8 activation bytes to [a0..a7, a0..a7] in BOTH 128-bit lanes.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<sbyte> TileAct(sbyte* a)
        {
            var bits = Unsafe.ReadUnaligned<long>(a);
            var v = Vector128.Create(bits, bits).AsSByte();
            return Vector256.Create(v, v);
        }

        // p03 = [r0:4|r1:4|r2:4|r3:4] int16 pair-sums, p47 likewise → 8 int32 lanes [r0..r7].
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<int> ReduceRows(
            Vector256<short> p03, Vector256<short> p47, Vector256<short> ones, Vector256<int> reduce)
        {
            var m03 = Avx2.MultiplyAddAdjacent(p03, ones); // [r0a,r0b,r1a,r1b|r2a,r2b,r3a,r3b]
            var m47 = Avx2.MultiplyAddAdjacent(p47, ones);
            var h = Avx2.HorizontalAdd(m03, m47);          // [r0,r1,r4,r5,r2,r3,r6,r7]
            return Avx2.PermuteVar8x32(h, reduce);         // [r0,r1,r2,r3,r4,r5,r6,r7]
        }

        // 8 lanes of SIGNED int8 scales at [base..base+8].
        private static unsafe Vector256<int> ScaleVec(byte* sc) =>
            Avx2.ConvertToVector256Int32(Vector128.CreateScalar(Unsafe.ReadUnaligned<long>(sc)).AsSByte());

        private static unsafe Vector256<float> LoadF16x8Int(byte* blk)
        {
            var u = (ushort*)blk;
            return Vector256.Create(
                (float)BitConverter.UInt16BitsToHalf(u[0]), (float)BitConverter.UInt16BitsToHalf(u[1]),
                (float)BitConverter.UInt16BitsToHalf(u[2]), (float)BitConverter.UInt16BitsToHalf(u[3]),
                (float)BitConverter.UInt16BitsToHalf(u[4]), (float)BitConverter.UInt16BitsToHalf(u[5]),
                (float)BitConverter.UInt16BitsToHalf(u[6]), (float)BitConverter.UInt16BitsToHalf(u[7]));
        }
    }
}
