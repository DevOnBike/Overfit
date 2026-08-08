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
        /// unpacked quants <i>in registers</i> and amortises them across columns instead.</para>
        ///
        /// <para>This sentence used to end "— which is exactly why the Q4_K tiled kernel measures ~3.3× over
        /// its own weight-stationary variant", a number copied here from the Q4_K kernel with the wrong
        /// baseline attached. Q4_K tiled versus Q4_K weight-stationary is an <b>exact tie (0.999×)</b>; the
        /// ~3× belongs to a comparison against re-decode-per-row. The Q6_K argument above stands on its own
        /// 13.5% measurement and never needed the borrowed one. Corrected 2026-08-07.</para>
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
            ReadOnlySpan<float> decodedScales = default,
            int groupStart = 0,
            int groupCount = 0)
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
#pragma warning disable OVERFIT026 // BOUND: cols is validated to [1, MaxTileCols=16] by the throw at the top of this method. Worst case 2 spans x 16 x 32 B = 1024 B.
            Span<Vector256<float>> sumf = stackalloc Vector256<float>[cols];
            Span<Vector256<int>> iacc = stackalloc Vector256<int>[cols];
#pragma warning restore OVERFIT026

            fixed (byte* rep = repacked)
            fixed (sbyte* aqAll = actQuants)
            fixed (float* asc = actScales)
            fixed (float* outp = output)
            fixed (float* dsc = decodedScales) // null when the caller did not pre-decode; see DecodeBlockScales
            {
                // Absolute group index, so a caller can hand this kernel one band of output rows and every
                // weight/output offset below still lands in the right place.
                var totalGroups = outputSize / 8;
                var groupEnd = groupCount <= 0 ? totalGroups : Math.Min(groupStart + groupCount, totalGroups);

                for (var x = groupStart; x < groupEnd; x++)
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
        /// AVX-512 form of <see cref="GemmTiled"/> that widens along the <b>weight layout</b> rather than
        /// across activation columns.
        ///
        /// <para><b>Why not column pairing.</b> The first attempt put column <c>2p</c> in the low half and
        /// <c>2p+1</c> in the high, which meant every weight vector had to be broadcast into both halves with a
        /// <c>vinserti64x4</c>. Q6_K issues six such broadcasts per <c>k</c>, sixteen times per block, for far
        /// less arithmetic each than Q4_K does — the lane-crossing traffic outran the arithmetic saved and the
        /// kernel measured <b>20% slower</b>. That version is recorded as a negative in ROADMAP.</para>
        ///
        /// <para><b>What this does instead.</b> <c>ql03</c> and <c>ql47</c> are already adjacent in the repacked
        /// block — at <c>k*64</c> and <c>k*64+32</c> — and so are <c>qhL03</c>/<c>qhL47</c>. One 512-bit load
        /// therefore carries <i>real data</i> in both halves, with no broadcast at all, and the halves stay
        /// independent all the way through <c>vpmaddubsw</c> and the subtract. Activations arrive as a single
        /// <c>vpbroadcastq</c> — <c>Vector512.Create(long)</c> replicates the 8-byte pattern eight times, which
        /// is exactly the tiling the 256-bit path built by hand.</para>
        ///
        /// <para>The only cross-half move left is one <c>GetUpper</c> per reduction, because AVX-512 provides no
        /// <c>vphaddd</c> for zmm. Accumulators stay 256-bit, so register pressure is unchanged.</para>
        ///
        /// <para><b>Bit-identical</b> to <see cref="GemmTiled"/>: the 512-bit operations act lanewise on exactly
        /// the values the two 256-bit operations did, and the horizontal add receives the same
        /// <c>(m03, m47)</c> pair it always did.</para>
        /// </summary>
        public static unsafe void GemmTiled512(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            int cols,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            Span<float> output,
            ReadOnlySpan<float> decodedScales = default,
            int groupStart = 0,
            int groupCount = 0)
        {
            if (cols is < 1 or > MaxTileCols)
            {
                throw new ArgumentOutOfRangeException(nameof(cols), cols, $"cols must be in [1, {MaxTileCols}].");
            }

            var nb = inputSize / 256;

            var m4b = Vector512.Create((byte)0x0F);
            var m2 = Vector512.Create((byte)0x03);
            var m32 = Vector512.Create((byte)32);
            var ones = Vector512.Create((short)1);
            var reduce = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);

#pragma warning disable OVERFIT026 // BOUND: cols is validated to [1, MaxTileCols=16] by the throw at the top of this method. Worst case 2 spans x 16 x 32 B = 1024 B.
            Span<Vector256<float>> sumf = stackalloc Vector256<float>[cols];
            Span<Vector256<int>> iacc = stackalloc Vector256<int>[cols];
#pragma warning restore OVERFIT026

            fixed (byte* rep = repacked)
            fixed (sbyte* aqAll = actQuants)
            fixed (float* asc = actScales)
            fixed (float* outp = output)
            fixed (float* dsc = decodedScales)
            {
                // Absolute group index, so a caller can hand this kernel one band of output rows and every
                // weight/output offset below still lands in the right place.
                var totalGroups = outputSize / 8;
                var groupEnd = groupCount <= 0 ? totalGroups : Math.Min(groupStart + groupCount, totalGroups);

                for (var x = groupStart; x < groupEnd; x++)
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

                            // One load each: rows 0-3 in the low half, rows 4-7 in the high half, as stored.
                            var ql0347 = Vector512.Load(ql + k * 64);
                            var qhL0347 = Vector512.Load(qh + qhBlockL);
                            var qhH0347 = Vector512.Load(qh + qhBlockH);

                            var qLu = LoNib512(ql0347, m4b) | QhBits512(qhL0347, qhShiftL, m2);
                            var qHu = HiNib512(ql0347, m4b) | QhBits512(qhH0347, qhShiftH, m2);

                            var scaleL = ScaleVec(scales + (baseL / 16) * 8);
                            var scaleH = ScaleVec(scales + (baseH / 16) * 8);

                            for (var c = 0; c < cols; c++)
                            {
                                var aqs = aqAll + (long)c * inputSize + l * 256;
                                var actL = TileAct512(aqs + baseL);
                                var actH = TileAct512(aqs + baseH);

                                var pL = Avx512BW.Subtract(Mul512(qLu, actL), Mul512(m32, actL));
                                var pH = Avx512BW.Subtract(Mul512(qHu, actH), Mul512(m32, actH));

                                var sumL = ReduceRows512(pL, ones, reduce);
                                var sumH = ReduceRows512(pH, ones, reduce);

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

        /// <summary>
        /// Eight activation bytes replicated across all 64 lanes by a single <c>vpbroadcastq</c> — the same
        /// tiling <see cref="TileAct"/> assembles from two Create calls, at no cross-half cost.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<sbyte> TileAct512(sbyte* a) =>
            Vector512.Create(Unsafe.ReadUnaligned<long>(a)).AsSByte();

        /// <summary>
        /// <see cref="ReduceRows"/> over a vector holding <c>[p03 | p47]</c>. The <c>vpmaddwd</c> runs at 512
        /// bits; the horizontal add then takes the two halves, which is exactly the <c>(m03, m47)</c> pair the
        /// 256-bit version passes, so the result is bit-identical.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<int> ReduceRows512(
            Vector512<short> p0347, Vector512<short> ones, Vector256<int> reduce)
        {
            var m = Avx512BW.MultiplyAddAdjacent(p0347, ones);

            return Avx2.PermuteVar8x32(Avx2.HorizontalAdd(m.GetLower(), m.GetUpper()), reduce);
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
