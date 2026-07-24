// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Repacked 8×8 Q4_K GEMV — produces 8 output rows at once from a
    /// <see cref="Q4KRepack"/> <c>block_q4_Kx8</c> weight and a Q8_K activation, with the 8
    /// row dot-products accumulated in SIMD lanes (no per-row horizontal reduction).
    /// Faithful AVX2 port of llama.cpp <c>ggml_gemv_q4_K_8x8_q8_K</c> (arch/x86/repack.cpp).
    /// Validated bit-close (maxRelDiff ~1e-6) to <see cref="Q4KDotKernel"/>; ~2× faster
    /// per core, &gt;3× parallelised.
    /// </summary>
    public static unsafe class Q4KGemvKernel
    {
        private const int BlockKx8Bytes = Q4KRepack.BlockKx8Bytes; // 1152
        private const int DstScalesOffset = 32;
        private const int DstQsOffset = 128;

        /// <summary>
        /// Opt-in (<c>OVERFIT_REPACK_GEMV=1</c>) for the repacked 8×8 decode GEMV. Off by
        /// default — it allocates a repacked weight copy per Q4_K FFN tensor (adds RAM) and is
        /// AVX2-only. When on, the decode FFN gate/up projections route here.
        /// </summary>
        public static readonly bool Enabled = ResolveFlag(OverfitEnvironment.RepackGemv);

        /// <summary>
        /// Opt-in (<c>OVERFIT_REPACK_ATTN=1</c>) for the whole-matrix Q4_K attention decode path (M3): the
        /// Q and O projections run as one repacked 8×8 GEMV over all heads (split per head AFTER), replacing
        /// the per-head Q4_K projections (measured 2.55× ‖ per projection). Off by default — it repacks the
        /// whole Q/O attention tensors (adds RAM) and reassociates the reduction (not bit-identical), so it is
        /// validated by E2E coherence, not byte-parity. AVX2-only. K stays per-head (cheap under GQA), V stays
        /// per-head (Q6_K under Q4_K_M).
        /// </summary>
        public static readonly bool AttnEnabled = ResolveFlag(OverfitEnvironment.RepackAttn);

        /// <summary>
        /// Opt-in (<c>OVERFIT_TILED_PREFILL=1</c>) for the register-tiled Q4_K prefill GEMM (<see cref="GemmTiled"/>)
        /// in place of the weight-stationary kernel — measured ~3× per projection under real parallelism. Off by
        /// default: it repacks the weight (adds ~model RAM) and is AVX2-only.
        /// </summary>
        public static readonly bool TiledPrefillEnabled = ResolveFlag(OverfitEnvironment.TiledPrefill);

        /// <summary>
        /// Measurement-only ablations for <see cref="GemmTiled"/>, all default-off. Each replaces one piece of
        /// per-block work with a constant so its share of the kernel's runtime can be read off directly.
        ///
        /// <para><b>Why ablation rather than micro-benchmarks.</b> The kernel reaches 1.70 TFLOP/s against 4.64
        /// for its own arithmetic instruction mix, and the 2.7× residual is work the mix benchmark does not
        /// model. Timing that work in isolation would measure a synthetic harness; toggling it inside the real
        /// kernel measures its actual share. <b>Results are wrong while an ablation is on</b> — these are timing
        /// probes, never a production path.</para>
        ///
        /// <para>Caveat when reading the numbers: removing a computation also lets the JIT fold or hoist what
        /// depended on it, so an ablation is an <i>upper</i> bound on the removed work's cost.</para>
        /// </summary>
        internal static bool AblateF16Scales;

        /// <inheritdoc cref="AblateF16Scales"/>
        internal static bool AblateScaleUnpack;

        /// <inheritdoc cref="AblateF16Scales"/>
        internal static bool AblateNibbleUnpack;

        private static bool ResolveFlag(string envVar)
        {
            if (!CpuFeatures.HasAvx2)
            {
                return false;
            }

            var raw = Environment.GetEnvironmentVariable(envVar);
            return raw is "1" || string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Sequential full-matrix GEMV (one thread).</summary>
        public static void Gemv(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            ReadOnlySpan<short> actBsums,
            Span<float> output)
        {
            var nb = inputSize / 256;
            fixed (byte* w = repacked)
            fixed (sbyte* aq = actQuants)
            fixed (float* asc = actScales)
            fixed (short* ab = actBsums)
            fixed (float* o = output)
            {
                for (var x = 0; x < outputSize / 8; x++)
                {
                    ComputeGroup(w, aq, asc, ab, o, x, nb);
                }
            }
        }

        /// <summary>
        /// Parallel full-matrix GEMV — the output-row-groups are split across the decode
        /// dispatch (capped / spin-pool). The <c>block_q4_Kx8</c> layout is row-group-major,
        /// so each worker owns a disjoint contiguous slice of weights + outputs.
        /// </summary>
        public static void GemvParallel(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            ReadOnlySpan<short> actBsums,
            Span<float> output)
        {
            var nb = inputSize / 256;
            var groups = outputSize / 8;

            fixed (byte* w = repacked)
            fixed (sbyte* aq = actQuants)
            fixed (float* asc = actScales)
            fixed (short* ab = actBsums)
            fixed (float* o = output)
            {
                var ctx = new GemvContext
                {
                    W = w,
                    Aq = aq,
                    Asc = asc,
                    Ab = ab,
                    Output = o,
                    Nb = nb,
                };
                OverfitParallel.ForDecode(0, groups, &GroupChunk, &ctx);
            }
        }

        private struct GemvContext
        {
            public byte* W;
            public sbyte* Aq;
            public float* Asc;
            public short* Ab;
            public float* Output;
            public int Nb;
        }

        private static void GroupChunk(int start, int end, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<GemvContext>(context);
            for (var x = start; x < end; x++)
            {
                ComputeGroup(ctx.W, ctx.Aq, ctx.Asc, ctx.Ab, ctx.Output, x, ctx.Nb);
            }
        }

        /// <summary>One row-group (8 output rows) — the validated inner kernel.</summary>
        private static void ComputeGroup(byte* w, sbyte* aq, float* asc, short* ab, float* output, int x, int nb)
        {
            var m4b = Vector256.Create((byte)0x0F);
            var deltamask = Vector128.Create((byte)0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
            var scalemask = Vector128.Create((byte)0, 0, 4, 4, 1, 1, 5, 5, 2, 2, 6, 6, 3, 3, 7, 7);
            var finalpermute = Vector256.Create(0, 2, 4, 6, 1, 3, 5, 7);
            const uint kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;

            var bptr = w + (long)x * nb * BlockKx8Bytes;
            var accRow = Vector256<float>.Zero;
            var accMin = Vector256<float>.Zero;

            // Scratch for the unpacked 6-bit scales/mins, hoisted out of BOTH loops (CA2014:
            // stackalloc in a loop grows the frame each iteration). Fully overwritten per use.
            var u0 = stackalloc uint[4];
            var u1 = stackalloc uint[4];

            for (var b = 0; b < nb; b++)
            {
                var blk = bptr + (long)b * BlockKx8Bytes;

                var rowScale = Vector256.Create(asc[b]);
                var colScale = LoadF16x8Rearrange(blk, deltamask);
                var colDmin = LoadF16x8(blk + 16);

                var q8sums = Vector256.Load(ab + b * 16);
                var q8sHadd = Ssse3.HorizontalAdd(q8sums.GetLower(), q8sums.GetUpper());
                var q8s = Vector256.Create(q8sHadd, q8sHadd).AsInt16();

                var iaccB = Vector256<int>.Zero;
                var iaccMinB = Vector256<int>.Zero;

                var qsBase = blk + DstQsOffset;
                var scBase = blk + DstScalesOffset;

                for (var sb = 0; sb < 4; sb++)
                {
                    var qs = qsBase + sb * 256;
                    var raw0123_0 = Vector256.Load(qs);
                    var raw4567_0 = Vector256.Load(qs + 32);
                    var raw0123_1 = Vector256.Load(qs + 64);
                    var raw4567_1 = Vector256.Load(qs + 96);
                    var raw0123_2 = Vector256.Load(qs + 128);
                    var raw4567_2 = Vector256.Load(qs + 160);
                    var raw0123_3 = Vector256.Load(qs + 192);
                    var raw4567_3 = Vector256.Load(qs + 224);

                    var v0123_00 = Avx2.And(raw0123_0, m4b);
                    var v4567_00 = Avx2.And(raw4567_0, m4b);
                    var v0123_01 = Avx2.And(raw0123_1, m4b);
                    var v4567_01 = Avx2.And(raw4567_1, m4b);
                    var v0123_02 = Avx2.And(raw0123_2, m4b);
                    var v4567_02 = Avx2.And(raw4567_2, m4b);
                    var v0123_03 = Avx2.And(raw0123_3, m4b);
                    var v4567_03 = Avx2.And(raw4567_3, m4b);

                    var v0123_10 = Avx2.And(Hi(raw0123_0), m4b);
                    var v4567_10 = Avx2.And(Hi(raw4567_0), m4b);
                    var v0123_11 = Avx2.And(Hi(raw0123_1), m4b);
                    var v4567_11 = Avx2.And(Hi(raw4567_1), m4b);
                    var v0123_12 = Avx2.And(Hi(raw0123_2), m4b);
                    var v4567_12 = Avx2.And(Hi(raw4567_2), m4b);
                    var v0123_13 = Avx2.And(Hi(raw0123_3), m4b);
                    var v4567_13 = Avx2.And(Hi(raw4567_3), m4b);

                    u0[0] = Unsafe.ReadUnaligned<uint>(scBase + 24 * sb);
                    u0[1] = Unsafe.ReadUnaligned<uint>(scBase + 24 * sb + 4);
                    u0[2] = Unsafe.ReadUnaligned<uint>(scBase + 24 * sb + 8);
                    u1[0] = Unsafe.ReadUnaligned<uint>(scBase + 12 + sb * 24);
                    u1[1] = Unsafe.ReadUnaligned<uint>(scBase + 12 + sb * 24 + 4);
                    u1[2] = Unsafe.ReadUnaligned<uint>(scBase + 12 + sb * 24 + 8);
                    Unpack(u0, kmask1, kmask2, kmask3);
                    Unpack(u1, kmask1, kmask2, kmask3);

                    var ms0 = Vector128.Create(u0[0], u0[1], u0[2], u0[3]).AsByte();
                    var ms1 = Vector128.Create(u1[0], u1[1], u1[2], u1[3]).AsByte();
                    var scales0 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(ms0, scalemask));
                    var scales1 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(ms1, scalemask));
                    var mins01 = Avx2.ConvertToVector256Int16(
                        Sse2.UnpackLow(
                            Sse2.Shuffle(ms0.AsInt32(), 78).AsByte(),
                            Sse2.Shuffle(ms1.AsInt32(), 78).AsByte()));

                    var aqb = aq + b * 256 + sb * 64;
                    var l00 = BroadcastLo(aqb);
                    var l01 = BroadcastLo(aqb + 16);
                    var l10 = BroadcastLo(aqb + 32);
                    var l11 = BroadcastLo(aqb + 48);

                    var iacc0 = Vector256<short>.Zero;
                    var iacc1 = Vector256<short>.Zero;

                    iacc0 = Avx2.Add(iacc0, Mul(Blend(v0123_00, Sh(v4567_00, 177)), Sh32(l00, 0)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(Sh(v0123_00, 177), v4567_00), Sh32(l00, 85)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(v0123_01, Sh(v4567_01, 177)), Sh32(l00, 170)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(Sh(v0123_01, 177), v4567_01), Sh32(l00, 255)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(v0123_02, Sh(v4567_02, 177)), Sh32(l01, 0)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(Sh(v0123_02, 177), v4567_02), Sh32(l01, 85)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(v0123_03, Sh(v4567_03, 177)), Sh32(l01, 170)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(Sh(v0123_03, 177), v4567_03), Sh32(l01, 255)));
                    var iacc0i = Avx2.MultiplyAddAdjacent(iacc0, scales0);

                    iacc1 = Avx2.Add(iacc1, Mul(Blend(v0123_10, Sh(v4567_10, 177)), Sh32(l10, 0)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(Sh(v0123_10, 177), v4567_10), Sh32(l10, 85)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(v0123_11, Sh(v4567_11, 177)), Sh32(l10, 170)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(Sh(v0123_11, 177), v4567_11), Sh32(l10, 255)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(v0123_12, Sh(v4567_12, 177)), Sh32(l11, 0)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(Sh(v0123_12, 177), v4567_12), Sh32(l11, 85)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(v0123_13, Sh(v4567_13, 177)), Sh32(l11, 170)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(Sh(v0123_13, 177), v4567_13), Sh32(l11, 255)));
                    var iacc1i = Avx2.MultiplyAddAdjacent(iacc1, scales1);

                    var iaccSb = Avx2.Add(iacc0i, iacc1i);

                    var q8sSb = Avx2.Shuffle(q8s.AsInt32(), 0).AsInt16();
                    var iaccMinSb = Avx2.MultiplyAddAdjacent(q8sSb, mins01);
                    q8s = Avx2.ShiftRightLogical128BitLane(q8s.AsByte(), 4).AsInt16();

                    iaccB = Avx2.Add(iaccB, iaccSb);
                    iaccMinB = Avx2.Add(iaccMinB, iaccMinSb);
                }

                accRow = Fma.MultiplyAdd(
                    Avx.ConvertToVector256Single(iaccB), Avx.Multiply(colScale, rowScale), accRow);
                accMin = Fma.MultiplyAdd(
                    Avx.ConvertToVector256Single(iaccMinB), Avx.Multiply(colDmin, rowScale), accMin);
            }

            accRow = Avx2.PermuteVar8x32(accRow, finalpermute);
            Avx.Subtract(accRow, accMin).Store(output + x * 8);
        }

        /// <summary>Floats written per weight block by <see cref="DecodeBlockScales"/>: 8 scales then 8 mins.</summary>
        public const int DecodedScalesPerBlock = 16;

        /// <summary>
        /// Decodes every weight block's F16 scale/min pair to <see cref="float"/> once, into
        /// <paramref name="destination"/> laid out as <c>[(group·nb + block) · 16]</c> — eight rearranged
        /// scales followed by eight mins.
        ///
        /// <para><b>Why this exists.</b> <see cref="GemmTiled"/> decodes these inline, once per (group, block).
        /// That looks amortised, but the kernel is invoked once per <i>column tile</i> — 84 times for a
        /// 672-token prompt at NR=8 — so the same F16 pairs are decoded 84 times over. Ablating the decode out
        /// of the kernel measured <b>12%</b> of its runtime, the largest single non-arithmetic item found.
        /// Hoisting it here reduces that work by the tile count instead of removing capability.</para>
        ///
        /// <para>Bit-identical to the inline path: the same conversions produce the same values, only fewer
        /// times. The alternative — widening <c>block_q4_Kx8</c> to hold f32 scales — costs 2.8% weight RAM
        /// permanently and invalidates every <c>.gguf.repack</c> sidecar; this costs a pooled scratch buffer
        /// that lives for one projection.</para>
        /// </summary>
        public static void DecodeBlockScales(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            Span<float> destination)
        {
            var nb = inputSize / 256;
            var groups = outputSize / 8;
            var deltamask = Vector128.Create((byte)0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);

            fixed (byte* w = repacked)
            fixed (float* d = destination)
            {
                for (var x = 0; x < groups; x++)
                {
                    for (var b = 0; b < nb; b++)
                    {
                        var index = ((long)x * nb + b) * DecodedScalesPerBlock;
                        var blk = w + ((long)x * nb + b) * BlockKx8Bytes;

                        LoadF16x8Rearrange(blk, deltamask).Store(d + index);
                        LoadF16x8(blk + 16).Store(d + index + 8);
                    }
                }
            }
        }

        /// <summary>Max activation columns per <see cref="GemmTiled"/> call — the register-tile width (NR). Kept
        /// small so the per-column accumulator scratch stays a bounded stack allocation; the caller tiles a
        /// larger prompt into NR-column chunks.</summary>
        public const int MaxTileCols = 16;

        /// <summary>
        /// Correctness-first register-tiled Q4_K GEMM (Phase 1 of the tinyBLAS lever): produces
        /// <paramref name="cols"/> output columns (activation vectors / prompt tokens) at once over the same
        /// <c>block_q4_Kx8</c> repacked weight, decoding each weight super-block ONCE and reusing it across all
        /// columns — the "second loop" the decode <see cref="Gemv"/> has nothing to tile. The per-(row,col)
        /// reduction order is identical to <see cref="Gemv"/>, so this is <b>bit-identical</b> to calling
        /// <see cref="Gemv"/> once per column (pinned by <c>Q4KTiledGemmParityTests</c>).
        ///
        /// <para>NOT yet perf-tuned: the per-column accumulators live in stack scratch (they spill), and the
        /// column loop is the plain inner loop. Phase 2 sweeps NR and lifts the hot accumulators into registers.
        /// Correctness first, performance second (a separate iteration A/B'd against this baseline).</para>
        ///
        /// Layout: activations are column-contiguous (column <c>c</c> owns <c>inputSize</c> quants,
        /// <c>inputSize/256</c> scales, <c>inputSize/16</c> bsums); output is column-major
        /// (<c>output[c*outputSize + row]</c>). AVX2.
        /// </summary>
        public static void GemmTiled(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            int cols,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            ReadOnlySpan<short> actBsums,
            Span<float> output,
            ReadOnlySpan<float> bias = default,
            int groupStart = 0,
            int groupCount = 0,
            ReadOnlySpan<float> decodedScales = default)
        {
            if (cols is < 1 or > MaxTileCols)
            {
                throw new ArgumentOutOfRangeException(nameof(cols), cols, $"cols must be in [1, {MaxTileCols}].");
            }

            if (!bias.IsEmpty && bias.Length < outputSize)
            {
                throw new ArgumentException(
                    $"bias length {bias.Length} < outputSize {outputSize}.", nameof(bias));
            }

            var nb = inputSize / 256;

            // Per-column accumulator state, hoisted out of the group loop (stackalloc-in-loop = CA2014). Reset
            // per output-group / per-block below. cols <= MaxTileCols keeps this a small bounded frame.
#pragma warning disable OVERFIT026 // BOUND: cols is validated to [1, MaxTileCols=16] by the throw at the top of this method. Worst case 5 spans x 16 x 32 B = 2560 B.
            Span<Vector256<float>> accRow = stackalloc Vector256<float>[cols];
            Span<Vector256<float>> accMin = stackalloc Vector256<float>[cols];
            Span<Vector256<int>> iaccB = stackalloc Vector256<int>[cols];
            Span<Vector256<int>> iaccMinB = stackalloc Vector256<int>[cols];
            Span<Vector256<short>> q8s = stackalloc Vector256<short>[cols];
#pragma warning restore OVERFIT026

            var m4b = Vector256.Create((byte)0x0F);
            var deltamask = Vector128.Create((byte)0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
            var scalemask = Vector128.Create((byte)0, 0, 4, 4, 1, 1, 5, 5, 2, 2, 6, 6, 3, 3, 7, 7);
            var finalpermute = Vector256.Create(0, 2, 4, 6, 1, 3, 5, 7);
            const uint kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;

            var u0 = stackalloc uint[4];
            var u1 = stackalloc uint[4];

            fixed (byte* w = repacked)
            fixed (sbyte* aq = actQuants)
            fixed (float* asc = actScales)
            fixed (short* ab = actBsums)
            fixed (float* o = output)
            fixed (float* bs = bias) // null when empty — keeps the no-bias path branch-free per store
            fixed (float* ds = decodedScales) // null when the caller did not pre-decode; see DecodeBlockScales
            {
                // Absolute group index throughout, so a caller can hand this kernel one band of output rows
                // and every weight/output/bias offset below still lands in the right place.
                var totalGroups = outputSize / 8;
                var groupEnd = groupCount <= 0
                    ? totalGroups
                    : Math.Min(groupStart + groupCount, totalGroups);

                for (var x = groupStart; x < groupEnd; x++)
                {
                    var bptr = w + (long)x * nb * BlockKx8Bytes;
                    for (var c = 0; c < cols; c++)
                    {
                        accRow[c] = Vector256<float>.Zero;
                        accMin[c] = Vector256<float>.Zero;
                    }

                    for (var b = 0; b < nb; b++)
                    {
                        var blk = bptr + (long)b * BlockKx8Bytes;

                        // Pre-decoded when the caller hoisted the F16 widening out of the tile loop; the values
                        // are identical either way, so the two paths are bit-identical.
                        var decodedAt = ds + (((long)x * nb) + b) * DecodedScalesPerBlock;
                        var colScale = ds is not null
                            ? Vector256.Load(decodedAt)
                            : AblateF16Scales ? Vector256.Create(1f) : LoadF16x8Rearrange(blk, deltamask);
                        var colDmin = ds is not null
                            ? Vector256.Load(decodedAt + 8)
                            : AblateF16Scales ? Vector256.Create(0f) : LoadF16x8(blk + 16);

                        var qsBase = blk + DstQsOffset;
                        var scBase = blk + DstScalesOffset;

                        // Per-column min-sum vector for this block (activation bsums), reset the integer accs.
                        for (var c = 0; c < cols; c++)
                        {
                            iaccB[c] = Vector256<int>.Zero;
                            iaccMinB[c] = Vector256<int>.Zero;
                            var q8sums = Vector256.Load(ab + (long)c * nb * 16 + b * 16);
                            var q8sHadd = Ssse3.HorizontalAdd(q8sums.GetLower(), q8sums.GetUpper());
                            q8s[c] = Vector256.Create(q8sHadd, q8sHadd).AsInt16();
                        }

                        for (var sb = 0; sb < 4; sb++)
                        {
                            var qs = qsBase + sb * 256;
                            var raw0123_0 = Vector256.Load(qs);
                            var raw4567_0 = Vector256.Load(qs + 32);
                            var raw0123_1 = Vector256.Load(qs + 64);
                            var raw4567_1 = Vector256.Load(qs + 96);
                            var raw0123_2 = Vector256.Load(qs + 128);
                            var raw4567_2 = Vector256.Load(qs + 160);
                            var raw0123_3 = Vector256.Load(qs + 192);
                            var raw4567_3 = Vector256.Load(qs + 224);

                            // Weight nibbles — decoded ONCE, reused across all cols (the tiling win).
                            var ablateNibbles = AblateNibbleUnpack;

                            var v0123_00 = ablateNibbles ? raw0123_0 : Avx2.And(raw0123_0, m4b);
                            var v4567_00 = ablateNibbles ? raw4567_0 : Avx2.And(raw4567_0, m4b);
                            var v0123_01 = ablateNibbles ? raw0123_1 : Avx2.And(raw0123_1, m4b);
                            var v4567_01 = ablateNibbles ? raw4567_1 : Avx2.And(raw4567_1, m4b);
                            var v0123_02 = ablateNibbles ? raw0123_2 : Avx2.And(raw0123_2, m4b);
                            var v4567_02 = ablateNibbles ? raw4567_2 : Avx2.And(raw4567_2, m4b);
                            var v0123_03 = ablateNibbles ? raw0123_3 : Avx2.And(raw0123_3, m4b);
                            var v4567_03 = ablateNibbles ? raw4567_3 : Avx2.And(raw4567_3, m4b);

                            var v0123_10 = ablateNibbles ? raw0123_0 : Avx2.And(Hi(raw0123_0), m4b);
                            var v4567_10 = ablateNibbles ? raw4567_0 : Avx2.And(Hi(raw4567_0), m4b);
                            var v0123_11 = ablateNibbles ? raw0123_1 : Avx2.And(Hi(raw0123_1), m4b);
                            var v4567_11 = ablateNibbles ? raw4567_1 : Avx2.And(Hi(raw4567_1), m4b);
                            var v0123_12 = ablateNibbles ? raw0123_2 : Avx2.And(Hi(raw0123_2), m4b);
                            var v4567_12 = ablateNibbles ? raw4567_2 : Avx2.And(Hi(raw4567_2), m4b);
                            var v0123_13 = ablateNibbles ? raw0123_3 : Avx2.And(Hi(raw0123_3), m4b);
                            var v4567_13 = ablateNibbles ? raw4567_3 : Avx2.And(Hi(raw4567_3), m4b);

                            u0[0] = Unsafe.ReadUnaligned<uint>(scBase + 24 * sb);
                            u0[1] = Unsafe.ReadUnaligned<uint>(scBase + 24 * sb + 4);
                            u0[2] = Unsafe.ReadUnaligned<uint>(scBase + 24 * sb + 8);
                            u1[0] = Unsafe.ReadUnaligned<uint>(scBase + 12 + sb * 24);
                            u1[1] = Unsafe.ReadUnaligned<uint>(scBase + 12 + sb * 24 + 4);
                            u1[2] = Unsafe.ReadUnaligned<uint>(scBase + 12 + sb * 24 + 8);

                            if (!AblateScaleUnpack)
                            {
                                Unpack(u0, kmask1, kmask2, kmask3);
                                Unpack(u1, kmask1, kmask2, kmask3);
                            }

                            var ms0 = Vector128.Create(u0[0], u0[1], u0[2], u0[3]).AsByte();
                            var ms1 = Vector128.Create(u1[0], u1[1], u1[2], u1[3]).AsByte();
                            var scales0 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(ms0, scalemask));
                            var scales1 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(ms1, scalemask));
                            var mins01 = Avx2.ConvertToVector256Int16(
                                Sse2.UnpackLow(
                                    Sse2.Shuffle(ms0.AsInt32(), 78).AsByte(),
                                    Sse2.Shuffle(ms1.AsInt32(), 78).AsByte()));

                            for (var c = 0; c < cols; c++)
                            {
                                var aqb = aq + (long)c * inputSize + b * 256 + sb * 64;
                                var l00 = BroadcastLo(aqb);
                                var l01 = BroadcastLo(aqb + 16);
                                var l10 = BroadcastLo(aqb + 32);
                                var l11 = BroadcastLo(aqb + 48);

                                var iacc0 = Vector256<short>.Zero;
                                var iacc1 = Vector256<short>.Zero;

                                iacc0 = Avx2.Add(iacc0, Mul(Blend(v0123_00, Sh(v4567_00, 177)), Sh32(l00, 0)));
                                iacc0 = Avx2.Add(iacc0, Mul(Blend(Sh(v0123_00, 177), v4567_00), Sh32(l00, 85)));
                                iacc0 = Avx2.Add(iacc0, Mul(Blend(v0123_01, Sh(v4567_01, 177)), Sh32(l00, 170)));
                                iacc0 = Avx2.Add(iacc0, Mul(Blend(Sh(v0123_01, 177), v4567_01), Sh32(l00, 255)));
                                iacc0 = Avx2.Add(iacc0, Mul(Blend(v0123_02, Sh(v4567_02, 177)), Sh32(l01, 0)));
                                iacc0 = Avx2.Add(iacc0, Mul(Blend(Sh(v0123_02, 177), v4567_02), Sh32(l01, 85)));
                                iacc0 = Avx2.Add(iacc0, Mul(Blend(v0123_03, Sh(v4567_03, 177)), Sh32(l01, 170)));
                                iacc0 = Avx2.Add(iacc0, Mul(Blend(Sh(v0123_03, 177), v4567_03), Sh32(l01, 255)));
                                var iacc0i = Avx2.MultiplyAddAdjacent(iacc0, scales0);

                                iacc1 = Avx2.Add(iacc1, Mul(Blend(v0123_10, Sh(v4567_10, 177)), Sh32(l10, 0)));
                                iacc1 = Avx2.Add(iacc1, Mul(Blend(Sh(v0123_10, 177), v4567_10), Sh32(l10, 85)));
                                iacc1 = Avx2.Add(iacc1, Mul(Blend(v0123_11, Sh(v4567_11, 177)), Sh32(l10, 170)));
                                iacc1 = Avx2.Add(iacc1, Mul(Blend(Sh(v0123_11, 177), v4567_11), Sh32(l10, 255)));
                                iacc1 = Avx2.Add(iacc1, Mul(Blend(v0123_12, Sh(v4567_12, 177)), Sh32(l11, 0)));
                                iacc1 = Avx2.Add(iacc1, Mul(Blend(Sh(v0123_12, 177), v4567_12), Sh32(l11, 85)));
                                iacc1 = Avx2.Add(iacc1, Mul(Blend(v0123_13, Sh(v4567_13, 177)), Sh32(l11, 170)));
                                iacc1 = Avx2.Add(iacc1, Mul(Blend(Sh(v0123_13, 177), v4567_13), Sh32(l11, 255)));
                                var iacc1i = Avx2.MultiplyAddAdjacent(iacc1, scales1);

                                var iaccSb = Avx2.Add(iacc0i, iacc1i);

                                var q8sSb = Avx2.Shuffle(q8s[c].AsInt32(), 0).AsInt16();
                                var iaccMinSb = Avx2.MultiplyAddAdjacent(q8sSb, mins01);
                                q8s[c] = Avx2.ShiftRightLogical128BitLane(q8s[c].AsByte(), 4).AsInt16();

                                iaccB[c] = Avx2.Add(iaccB[c], iaccSb);
                                iaccMinB[c] = Avx2.Add(iaccMinB[c], iaccMinSb);
                            }
                        }

                        for (var c = 0; c < cols; c++)
                        {
                            var rowScale = Vector256.Create(asc[(long)c * nb + b]);
                            accRow[c] = Fma.MultiplyAdd(
                                Avx.ConvertToVector256Single(iaccB[c]), Avx.Multiply(colScale, rowScale), accRow[c]);
                            accMin[c] = Fma.MultiplyAdd(
                                Avx.ConvertToVector256Single(iaccMinB[c]), Avx.Multiply(colDmin, rowScale), accMin[c]);
                        }
                    }

                    // Two stores rather than one with a zero vector: `x + 0f` rewrites -0.0 to +0.0,
                    // which would break the bit-identity the no-bias path is pinned to. The branch is
                    // per output-group, not per column, and is perfectly predicted.
                    if (bs is null)
                    {
                        for (var c = 0; c < cols; c++)
                        {
                            var row = Avx2.PermuteVar8x32(accRow[c], finalpermute);
                            Avx.Subtract(row, accMin[c]).Store(o + (long)c * outputSize + x * 8);
                        }
                    }

                    if (bs is not null)
                    {
                        // Same 8 bias floats for every column - hoisted out of the column loop.
                        var biasVec = Vector256.Load(bs + x * 8);
                        for (var c = 0; c < cols; c++)
                        {
                            var row = Avx2.PermuteVar8x32(accRow[c], finalpermute);
                            Avx.Add(Avx.Subtract(row, accMin[c]), biasVec)
                                .Store(o + (long)c * outputSize + x * 8);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// AVX-512 form of <see cref="GemmTiled"/>: identical arithmetic, but <b>two activation columns per
        /// instruction</b> — column <c>2p</c> in the low 256 bits of every vector, column <c>2p+1</c> in the
        /// high 256. The weights are the same for both, so they are broadcast into both halves; only the
        /// activations, their scales and their block sums differ per half.
        ///
        /// <para><b>Why columns and not output rows.</b> Widening the output-row group to 16 would need a new
        /// <c>block_q4_Kx16</c> repack layout and would invalidate every sidecar. Pairing columns reuses
        /// <c>block_q4_Kx8</c> untouched, and every shuffle in this kernel is per-128-bit-lane, so it widens
        /// without changing meaning.</para>
        ///
        /// <para><b>Why the pair loop stays innermost.</b> Hoisting it would re-decode the sixteen weight
        /// vectors per pair. Amortising that fixed per-block work across the whole column tile is precisely
        /// what the tile-width sweep showed to dominate this kernel, so the loop order is preserved exactly.</para>
        ///
        /// <para><b>Bit-identical</b> to <see cref="GemmTiled"/>: each column's operations and their order are
        /// unchanged, two columns merely execute at once. Measured ceilings on this machine: the kernel's own
        /// instruction mix runs at 4.63 TFLOP/s at 256 bits and 7.71–9.08 at 512.</para>
        /// </summary>
        public static void GemmTiled512(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            int cols,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            ReadOnlySpan<short> actBsums,
            Span<float> output,
            ReadOnlySpan<float> bias = default,
            ReadOnlySpan<float> decodedScales = default)
        {
            if (cols is < 1 or > MaxTileCols)
            {
                throw new ArgumentOutOfRangeException(nameof(cols), cols, $"cols must be in [1, {MaxTileCols}].");
            }

            if (!bias.IsEmpty && bias.Length < outputSize)
            {
                throw new ArgumentException(
                    $"bias length {bias.Length} < outputSize {outputSize}.", nameof(bias));
            }

            var nb = inputSize / 256;
            var pairs = (cols + 1) / 2;

#pragma warning disable OVERFIT026 // BOUND: pairs = (cols + 1) / 2 and cols is validated to [1, MaxTileCols=16] above, so pairs <= 8. Worst case 5 spans x 8 x 64 B = 2560 B.
            Span<Vector512<float>> accRow = stackalloc Vector512<float>[pairs];
            Span<Vector512<float>> accMin = stackalloc Vector512<float>[pairs];
            Span<Vector512<int>> iaccB = stackalloc Vector512<int>[pairs];
            Span<Vector512<int>> iaccMinB = stackalloc Vector512<int>[pairs];
            Span<Vector512<short>> q8s = stackalloc Vector512<short>[pairs];
#pragma warning restore OVERFIT026

            var m4b = Vector512.Create((byte)0x0F);
            var deltamask = Vector128.Create((byte)0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
            var scalemask = Vector128.Create((byte)0, 0, 4, 4, 1, 1, 5, 5, 2, 2, 6, 6, 3, 3, 7, 7);
            var finalpermute = Vector256.Create(0, 2, 4, 6, 1, 3, 5, 7);

            // Avx2.Blend(..., 170) takes the odd int32 lanes from the right operand; over sixteen lanes that is
            // the same alternating pattern, expressed as a mask vector because AVX-512 blends by mask.
            var blendMask = Vector512.Create(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1);

            const uint kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;

            var u0 = stackalloc uint[4];
            var u1 = stackalloc uint[4];

            fixed (byte* w = repacked)
            fixed (sbyte* aq = actQuants)
            fixed (float* asc = actScales)
            fixed (short* ab = actBsums)
            fixed (float* o = output)
            fixed (float* bs = bias)
            fixed (float* ds = decodedScales)
            {
                for (var x = 0; x < outputSize / 8; x++)
                {
                    var bptr = w + (long)x * nb * BlockKx8Bytes;

                    for (var p = 0; p < pairs; p++)
                    {
                        accRow[p] = Vector512<float>.Zero;
                        accMin[p] = Vector512<float>.Zero;
                    }

                    for (var b = 0; b < nb; b++)
                    {
                        var blk = bptr + (long)b * BlockKx8Bytes;
                        var decodedAt = ds + (((long)x * nb) + b) * DecodedScalesPerBlock;

                        var colScale256 = ds is not null
                            ? Vector256.Load(decodedAt)
                            : LoadF16x8Rearrange(blk, deltamask);
                        var colDmin256 = ds is not null
                            ? Vector256.Load(decodedAt + 8)
                            : LoadF16x8(blk + 16);

                        var colScale = Vector512.Create(colScale256, colScale256);
                        var colDmin = Vector512.Create(colDmin256, colDmin256);

                        var qsBase = blk + DstQsOffset;
                        var scBase = blk + DstScalesOffset;

                        for (var p = 0; p < pairs; p++)
                        {
                            iaccB[p] = Vector512<int>.Zero;
                            iaccMinB[p] = Vector512<int>.Zero;
                            q8s[p] = Vector512.Create(BlockSums(ab, PairLow(p), nb, b), BlockSums(ab, PairHigh(p, cols), nb, b));
                        }

                        for (var sb = 0; sb < 4; sb++)
                        {
                            var qs = qsBase + sb * 256;

                            var raw0123_0 = Broadcast512(qs);
                            var raw4567_0 = Broadcast512(qs + 32);
                            var raw0123_1 = Broadcast512(qs + 64);
                            var raw4567_1 = Broadcast512(qs + 96);
                            var raw0123_2 = Broadcast512(qs + 128);
                            var raw4567_2 = Broadcast512(qs + 160);
                            var raw0123_3 = Broadcast512(qs + 192);
                            var raw4567_3 = Broadcast512(qs + 224);

                            var v0123_00 = raw0123_0 & m4b;
                            var v4567_00 = raw4567_0 & m4b;
                            var v0123_01 = raw0123_1 & m4b;
                            var v4567_01 = raw4567_1 & m4b;
                            var v0123_02 = raw0123_2 & m4b;
                            var v4567_02 = raw4567_2 & m4b;
                            var v0123_03 = raw0123_3 & m4b;
                            var v4567_03 = raw4567_3 & m4b;

                            var v0123_10 = Hi512(raw0123_0) & m4b;
                            var v4567_10 = Hi512(raw4567_0) & m4b;
                            var v0123_11 = Hi512(raw0123_1) & m4b;
                            var v4567_11 = Hi512(raw4567_1) & m4b;
                            var v0123_12 = Hi512(raw0123_2) & m4b;
                            var v4567_12 = Hi512(raw4567_2) & m4b;
                            var v0123_13 = Hi512(raw0123_3) & m4b;
                            var v4567_13 = Hi512(raw4567_3) & m4b;

                            u0[0] = Unsafe.ReadUnaligned<uint>(scBase + 24 * sb);
                            u0[1] = Unsafe.ReadUnaligned<uint>(scBase + 24 * sb + 4);
                            u0[2] = Unsafe.ReadUnaligned<uint>(scBase + 24 * sb + 8);
                            u1[0] = Unsafe.ReadUnaligned<uint>(scBase + 12 + sb * 24);
                            u1[1] = Unsafe.ReadUnaligned<uint>(scBase + 12 + sb * 24 + 4);
                            u1[2] = Unsafe.ReadUnaligned<uint>(scBase + 12 + sb * 24 + 8);
                            Unpack(u0, kmask1, kmask2, kmask3);
                            Unpack(u1, kmask1, kmask2, kmask3);

                            var ms0 = Vector128.Create(u0[0], u0[1], u0[2], u0[3]).AsByte();
                            var ms1 = Vector128.Create(u1[0], u1[1], u1[2], u1[3]).AsByte();
                            var s0 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(ms0, scalemask));
                            var s1 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(ms1, scalemask));
                            var mn = Avx2.ConvertToVector256Int16(
                                Sse2.UnpackLow(
                                    Sse2.Shuffle(ms0.AsInt32(), 78).AsByte(),
                                    Sse2.Shuffle(ms1.AsInt32(), 78).AsByte()));

                            var scales0 = Vector512.Create(s0, s0);
                            var scales1 = Vector512.Create(s1, s1);
                            var mins01 = Vector512.Create(mn, mn);

                            for (var p = 0; p < pairs; p++)
                            {
                                var lowCol = PairLow(p);
                                var highCol = PairHigh(p, cols);
                                var aLow = aq + (long)lowCol * inputSize + b * 256 + sb * 64;
                                var aHigh = aq + (long)highCol * inputSize + b * 256 + sb * 64;

                                var l00 = BroadcastLo512(aLow, aHigh);
                                var l01 = BroadcastLo512(aLow + 16, aHigh + 16);
                                var l10 = BroadcastLo512(aLow + 32, aHigh + 32);
                                var l11 = BroadcastLo512(aLow + 48, aHigh + 48);

                                var iacc0 = Vector512<short>.Zero;
                                var iacc1 = Vector512<short>.Zero;

                                iacc0 = Avx512BW.Add(iacc0, Mul512(Blend512(v0123_00, Sh512(v4567_00, 177), blendMask), Sh32_512(l00, 0)));
                                iacc0 = Avx512BW.Add(iacc0, Mul512(Blend512(Sh512(v0123_00, 177), v4567_00, blendMask), Sh32_512(l00, 85)));
                                iacc0 = Avx512BW.Add(iacc0, Mul512(Blend512(v0123_01, Sh512(v4567_01, 177), blendMask), Sh32_512(l00, 170)));
                                iacc0 = Avx512BW.Add(iacc0, Mul512(Blend512(Sh512(v0123_01, 177), v4567_01, blendMask), Sh32_512(l00, 255)));
                                iacc0 = Avx512BW.Add(iacc0, Mul512(Blend512(v0123_02, Sh512(v4567_02, 177), blendMask), Sh32_512(l01, 0)));
                                iacc0 = Avx512BW.Add(iacc0, Mul512(Blend512(Sh512(v0123_02, 177), v4567_02, blendMask), Sh32_512(l01, 85)));
                                iacc0 = Avx512BW.Add(iacc0, Mul512(Blend512(v0123_03, Sh512(v4567_03, 177), blendMask), Sh32_512(l01, 170)));
                                iacc0 = Avx512BW.Add(iacc0, Mul512(Blend512(Sh512(v0123_03, 177), v4567_03, blendMask), Sh32_512(l01, 255)));
                                var iacc0i = Avx512BW.MultiplyAddAdjacent(iacc0, scales0);

                                iacc1 = Avx512BW.Add(iacc1, Mul512(Blend512(v0123_10, Sh512(v4567_10, 177), blendMask), Sh32_512(l10, 0)));
                                iacc1 = Avx512BW.Add(iacc1, Mul512(Blend512(Sh512(v0123_10, 177), v4567_10, blendMask), Sh32_512(l10, 85)));
                                iacc1 = Avx512BW.Add(iacc1, Mul512(Blend512(v0123_11, Sh512(v4567_11, 177), blendMask), Sh32_512(l10, 170)));
                                iacc1 = Avx512BW.Add(iacc1, Mul512(Blend512(Sh512(v0123_11, 177), v4567_11, blendMask), Sh32_512(l10, 255)));
                                iacc1 = Avx512BW.Add(iacc1, Mul512(Blend512(v0123_12, Sh512(v4567_12, 177), blendMask), Sh32_512(l11, 0)));
                                iacc1 = Avx512BW.Add(iacc1, Mul512(Blend512(Sh512(v0123_12, 177), v4567_12, blendMask), Sh32_512(l11, 85)));
                                iacc1 = Avx512BW.Add(iacc1, Mul512(Blend512(v0123_13, Sh512(v4567_13, 177), blendMask), Sh32_512(l11, 170)));
                                iacc1 = Avx512BW.Add(iacc1, Mul512(Blend512(Sh512(v0123_13, 177), v4567_13, blendMask), Sh32_512(l11, 255)));
                                var iacc1i = Avx512BW.MultiplyAddAdjacent(iacc1, scales1);

                                var q8sSb = Avx512F.Shuffle(q8s[p].AsInt32(), 0).AsInt16();
                                var iaccMinSb = Avx512BW.MultiplyAddAdjacent(q8sSb, mins01);
                                q8s[p] = Avx512BW.ShiftRightLogical128BitLane(q8s[p].AsByte(), 4).AsInt16();

                                iaccB[p] = Avx512F.Add(iaccB[p], Avx512F.Add(iacc0i, iacc1i));
                                iaccMinB[p] = Avx512F.Add(iaccMinB[p], iaccMinSb);
                            }
                        }

                        for (var p = 0; p < pairs; p++)
                        {
                            var rowScale = Vector512.Create(
                                Vector256.Create(asc[(long)PairLow(p) * nb + b]),
                                Vector256.Create(asc[(long)PairHigh(p, cols) * nb + b]));

                            accRow[p] = Avx512F.FusedMultiplyAdd(
                                Avx512F.ConvertToVector512Single(iaccB[p]),
                                Avx512F.Multiply(colScale, rowScale),
                                accRow[p]);
                            accMin[p] = Avx512F.FusedMultiplyAdd(
                                Avx512F.ConvertToVector512Single(iaccMinB[p]),
                                Avx512F.Multiply(colDmin, rowScale),
                                accMin[p]);
                        }
                    }

                    for (var p = 0; p < pairs; p++)
                    {
                        var lowCol = PairLow(p);
                        var highCol = 2 * p + 1;

                        StoreColumn(o, bs, accRow[p].GetLower(), accMin[p].GetLower(),
                            finalpermute, lowCol, outputSize, x);

                        // The odd tail computed a duplicate of the low column in the high half; discard it.
                        if (highCol < cols)
                        {
                            StoreColumn(o, bs, accRow[p].GetUpper(), accMin[p].GetUpper(),
                                finalpermute, highCol, outputSize, x);
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int PairLow(int pair) => 2 * pair;

        /// <summary>The odd column of a pair, or the even one again when the tile has an odd column count.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int PairHigh(int pair, int cols) => Math.Min(2 * pair + 1, cols - 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<short> BlockSums(short* ab, int column, int nb, int b)
        {
            var q8sums = Vector256.Load(ab + (long)column * nb * 16 + b * 16);
            var hadd = Ssse3.HorizontalAdd(q8sums.GetLower(), q8sums.GetUpper());

            return Vector256.Create(hadd, hadd).AsInt16();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StoreColumn(
            float* o,
            float* bs,
            Vector256<float> row,
            Vector256<float> min,
            Vector256<int> finalpermute,
            int column,
            int outputSize,
            int x)
        {
            var permuted = Avx2.PermuteVar8x32(row, finalpermute);
            var value = Avx.Subtract(permuted, min);

            // Two stores rather than adding a zero vector: `x + 0f` rewrites -0.0 to +0.0 and would break the
            // bit-identity the no-bias path is pinned to.
            if (bs is null)
            {
                value.Store(o + (long)column * outputSize + x * 8);
                return;
            }

            Avx.Add(value, Vector256.Load(bs + x * 8)).Store(o + (long)column * outputSize + x * 8);
        }

        /// <summary>The same 32 weight bytes in both halves — weights are shared by the two columns of a pair.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<byte> Broadcast512(byte* p)
        {
            var v = Vector256.Load(p);

            return Vector512.Create(v, v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<byte> Hi512(Vector512<byte> v) =>
            Avx512BW.ShiftRightLogical(v.AsUInt16(), 4).AsByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<byte> Sh512(Vector512<byte> v, [ConstantExpected] byte imm) =>
            Avx512F.Shuffle(v.AsInt32(), imm).AsByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<sbyte> Sh32_512(Vector512<sbyte> v, [ConstantExpected] byte imm) =>
            Avx512F.Shuffle(v.AsInt32(), imm).AsSByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<byte> Blend512(Vector512<byte> a, Vector512<byte> b, Vector512<int> mask) =>
            Avx512F.BlendVariable(a.AsInt32(), b.AsInt32(), mask).AsByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<short> Mul512(Vector512<byte> rhs, Vector512<sbyte> lhs) =>
            Avx512BW.MultiplyAddAdjacent(rhs, lhs);

        /// <summary>One 16-byte activation run per half, duplicated within each half exactly as <see cref="BroadcastLo"/> does.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<sbyte> BroadcastLo512(sbyte* low, sbyte* high)
        {
            var l = Vector128.Load(low);
            var h = Vector128.Load(high);

            return Vector512.Create(Vector256.Create(l, l), Vector256.Create(h, h));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> Hi(Vector256<byte> v) => Avx2.ShiftRightLogical(v.AsUInt16(), 4).AsByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> Sh(Vector256<byte> v, [ConstantExpected] byte imm) => Avx2.Shuffle(v.AsInt32(), imm).AsByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<sbyte> Sh32(Vector256<sbyte> v, [ConstantExpected] byte imm) => Avx2.Shuffle(v.AsInt32(), imm).AsSByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> Blend(Vector256<byte> a, Vector256<byte> b) =>
            Avx2.Blend(a.AsInt32(), b.AsInt32(), 170).AsByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<short> Mul(Vector256<byte> rhs, Vector256<sbyte> lhs) =>
            Avx2.MultiplyAddAdjacent(rhs, lhs);

        private static Vector256<sbyte> BroadcastLo(sbyte* p)
        {
            var lo = Vector128.Load(p);
            return Vector256.Create(lo, lo);
        }

        private static void Unpack(uint* u, uint k1, uint k2, uint k3)
        {
            u[3] = ((u[2] >> 4) & k2) | (((u[1] >> 6) & k3) << 4);
            var aux = u[1] & k1;
            u[1] = (u[2] & k2) | (((u[0] >> 6) & k3) << 4);
            u[2] = aux;
            u[0] &= k1;
        }

        /// <summary>
        /// Widens eight IEEE half-precision values to <see cref="float"/>.
        ///
        /// <para>The hardware path is one <c>vcvtph2ps</c>. The scalar fallback below costs eight
        /// <see cref="BitConverter.UInt16BitsToHalf"/> calls plus a <see cref="Vector256"/> build, and ablating
        /// this decode out of <c>GemmTiled</c> measured <b>12%</b> of the kernel's runtime — it runs once per
        /// weight super-block and so is not amortised across the column tile. Half→float is exact in both
        /// paths (no rounding is possible when widening), so the two are bit-identical.</para>
        /// </summary>
        private static Vector256<float> LoadF16x8(byte* p)
        {
            var u = (ushort*)p;
            return Vector256.Create(
                (float)BitConverter.UInt16BitsToHalf(u[0]), (float)BitConverter.UInt16BitsToHalf(u[1]),
                (float)BitConverter.UInt16BitsToHalf(u[2]), (float)BitConverter.UInt16BitsToHalf(u[3]),
                (float)BitConverter.UInt16BitsToHalf(u[4]), (float)BitConverter.UInt16BitsToHalf(u[5]),
                (float)BitConverter.UInt16BitsToHalf(u[6]), (float)BitConverter.UInt16BitsToHalf(u[7]));
        }

        /// <summary>
        /// The same widening as <see cref="LoadF16x8"/>, after the repacked layout's byte rearrangement.
        ///
        /// <para>The lanes are extracted straight out of the register with <c>pextrw</c>. The previous version
        /// stored the shuffled vector into a <c>stackalloc</c> buffer and immediately re-read it as eight
        /// <see cref="ushort"/>s — a 16-byte store followed by eight narrow loads of the same address, which
        /// is the pathological case for store-to-load forwarding: the loads cannot be satisfied from the store
        /// buffer and stall until the store retires to L1.</para>
        ///
        /// <para>x86 has <c>vcvtph2ps</c>, which would widen all eight in one instruction, but .NET exposes
        /// neither an <c>F16C</c> intrinsic class nor a <see cref="Half"/> overload of
        /// <see cref="Vector128.Widen(Vector128{float})"/>, so the conversions stay scalar. Ablating this decode
        /// out of the kernel measured 12% of its runtime; removing only the round-trip recovers whatever share
        /// of that was the stall rather than the arithmetic.</para>
        /// </summary>
        private static Vector256<float> LoadF16x8Rearrange(byte* p, Vector128<byte> deltamask)
        {
            var v = Ssse3.Shuffle(Vector128.Load(p), deltamask).AsUInt16();

            return Vector256.Create(
                (float)BitConverter.UInt16BitsToHalf(v.GetElement(0)),
                (float)BitConverter.UInt16BitsToHalf(v.GetElement(1)),
                (float)BitConverter.UInt16BitsToHalf(v.GetElement(2)),
                (float)BitConverter.UInt16BitsToHalf(v.GetElement(3)),
                (float)BitConverter.UInt16BitsToHalf(v.GetElement(4)),
                (float)BitConverter.UInt16BitsToHalf(v.GetElement(5)),
                (float)BitConverter.UInt16BitsToHalf(v.GetElement(6)),
                (float)BitConverter.UInt16BitsToHalf(v.GetElement(7)));
        }
    }
}
