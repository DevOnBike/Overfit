// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Kernels
{
    internal static unsafe class LinearKernels
    {
        private const int InputMajorVectorizedOutputThreshold = 32;

        public static void TransposeInputOutputToOutputInput(
            ReadOnlySpan<float> sourceInputOutput,
            Span<float> destinationOutputInput,
            int inputSize,
            int outputSize)
        {
            if (sourceInputOutput.Length < inputSize * outputSize)
            {
                throw new ArgumentException("Source weights span is too small.", nameof(sourceInputOutput));
            }

            if (destinationOutputInput.Length < inputSize * outputSize)
            {
                throw new ArgumentException("Destination weights span is too small.", nameof(destinationOutputInput));
            }

            for (var i = 0; i < inputSize; i++)
            {
                var srcBase = i * outputSize;

                for (var j = 0; j < outputSize; j++)
                {
                    destinationOutputInput[j * inputSize + i] = sourceInputOutput[srcBase + j];
                }
            }
        }

        public static void Forward(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> weightsOutputInput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputSize);

            if (input.Length % inputSize != 0)
            {
                throw new ArgumentException(
                    "Input length is not divisible by layer input size.",
                    nameof(input));
            }

            if (weightsInputOutput.Length < inputSize * outputSize)
            {
                throw new ArgumentException(
                    "Input-major weights span is too small.",
                    nameof(weightsInputOutput));
            }

            if (weightsOutputInput.Length < inputSize * outputSize)
            {
                throw new ArgumentException(
                    "Output-major weights span is too small.",
                    nameof(weightsOutputInput));
            }

            if (bias.Length < outputSize)
            {
                throw new ArgumentException(
                    "Bias span is too small.",
                    nameof(bias));
            }

            var batchSize = input.Length / inputSize;
            var expectedOutputLength = batchSize * outputSize;

            if (output.Length < expectedOutputLength)
            {
                throw new ArgumentException(
                    "Output span is too small for Linear inference.",
                    nameof(output));
            }

            // Bytes the whole call moves, which is the weight matrix once PER ROW — this kernel iterates rows
            // outermost and re-reads the weights for each. At batch 1 this is identical to the weight size,
            // so the crossover measured there still applies unchanged; above it, a small layer with a large
            // batch correctly qualifies on the traffic it actually generates rather than being judged on one
            // row's worth (`XC-79`).
            var parallelWork = (long)batchSize * inputSize * outputSize * sizeof(float);

            if (outputSize >= InputMajorVectorizedOutputThreshold
                && parallelWork >= ForwardParallelWeightBytes)
            {
                ForwardColumnsParallel(
                    input,
                    weightsInputOutput,
                    bias,
                    output,
                    batchSize,
                    inputSize,
                    outputSize);

                return;
            }

            // Narrow output plus more than one row: tile the batch against the outputs instead of running one
            // dot per output per row (`XC-80`). Gated on batchSize > 1 because a single row has nothing to
            // tile against, which keeps the published `Linear(784,10)` batch-1 figure on its original path.
            if (batchSize > 1 && outputSize < InputMajorVectorizedOutputThreshold)
            {
                ForwardOutputMajorTiled(
                    input,
                    weightsOutputInput,
                    bias,
                    output,
                    batchSize,
                    inputSize,
                    outputSize);

                return;
            }

            for (var b = 0; b < batchSize; b++)
            {
                var inSlice = input.Slice(b * inputSize, inputSize);
                var outSlice = output.Slice(b * outputSize, outputSize);

                if (outputSize >= InputMajorVectorizedOutputThreshold)
                {
                    ForwardInputMajorVector4(
                        inSlice,
                        weightsInputOutput,
                        bias,
                        outSlice,
                        inputSize,
                        outputSize,
                        0,
                        outputSize);

                    continue;
                }

                {
                    ForwardOutputMajorDot(
                        inSlice,
                        weightsOutputInput,
                        bias,
                        outSlice);
                }
            }
        }

        /// <summary>
        /// Weight bytes at or above which forward inference splits the output columns across workers.
        ///
        /// <para><b>The unit is BYTES READ, not FLOPs, because this kernel is bandwidth-bound at batch 1.</b>
        /// A <c>Linear(inputSize, outputSize)</c> forward reads the whole weight matrix once and does
        /// <c>2·inputSize·outputSize</c> flops against <c>4·inputSize·outputSize</c> bytes — half a flop per
        /// byte, which is two orders below this machine's balance point. So what decides whether extra cores
        /// help is whether the weight read is large enough to need more than one core's share of memory
        /// bandwidth, and a FLOP-based threshold would be measuring the wrong quantity (`XC-79`).</para>
        ///
        /// <para><b>Why a threshold at all, rather than always parallelising.</b> <c>OverfitParallel.For</c>
        /// costs single-digit microseconds to dispatch. <c>Linear(784, 10)</c> — this repository's published
        /// 8.3×-vs-ONNX-Runtime number — completes in <b>237 ns</b>, so dispatching it would be roughly a
        /// 10× regression on a figure that is in the README. The threshold exists to keep that path
        /// untouched.</para>
        ///
        /// <para><b>Measured, not chosen</b> — and the first value here was chosen, which is why this says so.
        /// 4 MB was picked a priori and landed on the one size the sweep says parallel LOSES. The sweep is
        /// <c>Tests/Diagnostics/LinearForwardParallelThresholdDiagnostics.cs</c>, run twice with this constant
        /// forced to <c>long.MaxValue</c> and to <c>0</c>:</para>
        ///
        /// <code>
        /// weights    serial     parallel   speedup
        ///   0.5 MB   0.0593 ms  0.0684 ms    0.87x
        ///   2   MB   0.0910 ms  0.0942 ms    0.97x
        ///   4   MB   0.1349 ms  0.2278 ms    0.59x
        ///   8   MB   0.3038 ms  0.2331 ms    1.30x
        ///  16   MB   0.4420 ms  0.2455 ms    1.80x
        /// 392   MB  31.44   ms 11.79   ms    2.67x
        /// </code>
        ///
        /// <para><b>The mechanism the numbers show, which is not the one expected.</b> The parallel arm costs
        /// ~0.22-0.25 ms at 4, 8 and 16 MB alike — four times the work for eight percent more time — so below
        /// that it is paying a fixed dispatch cost of roughly <b>0.22 ms</b>, not doing the work. The
        /// crossover is therefore wherever the serial time passes that floor, which lands at <b>8 MB</b> on
        /// this box, and it has nothing to do with FLOPs. Set to the first size measured to win rather than
        /// to the midpoint: at the crossover the two paths are equal, so the tie should go to the one with no
        /// dispatch and no threads.</para>
        ///
        /// <para><b>Machine-specific, and deliberately not auto-tuned.</b> A box with a cheaper thread wake or
        /// less memory bandwidth moves this. Re-run the sweep rather than reasoning about the constant.</para>
        /// </summary>
        internal const long ForwardParallelWeightBytes = 8L * 1024 * 1024;

        /// <summary>
        /// One inference row, with the output columns split across workers.
        ///
        /// <para>Splitting on <b>columns</b> rather than on the batch is what makes this safe without any
        /// synchronisation: each worker owns a disjoint slice of <paramref name="output"/>, and everything
        /// else it touches — the input vector and the weight matrix — is read-only. The input is re-read by
        /// every worker, which is free: it is one vector, and at batch 1 it is orders of magnitude smaller
        /// than the weights it is multiplied against.</para>
        ///
        /// <para><b>ONE dispatch for the whole call, not one per batch row.</b> The work is the rectangle
        /// <c>batchSize × outputSize</c> and it is dispatched flat, so a worker's range can start mid-row and
        /// end mid-row. That costs a little index arithmetic and buys two things a per-row dispatch cannot:
        /// the fixed dispatch cost is paid once instead of <c>batchSize</c> times, and every worker stays busy
        /// even when the batch is smaller than the pool — splitting on the batch alone would leave 30 of 32
        /// workers idle at batch 2.</para>
        ///
        /// <para><b>Measured, and the per-row shape was genuinely bad.</b> Before this, cost per row sat flat
        /// at ~0.23 ms for batch 1 through 8 — each row paying its own dispatch. Flattening the dispatch, on
        /// the 2048x1024 layer:</para>
        ///
        /// <code>
        /// batch   per row before   per row after
        ///     1       0.2338 ms       0.2484 ms   (unchanged: one row is one dispatch either way)
        ///     2       0.2382          0.1267      1.88x
        ///     4       0.2331          0.0671      3.47x
        ///     8       0.2332          0.0428      5.45x
        ///    32       0.0346          0.0222      1.56x
        ///    64       0.0349          0.0173      2.02x
        /// </code>
        ///
        /// <para>Per-row cost now falls monotonically as the batch grows, which is the shape one amortised
        /// dispatch predicts. <b>The old numbers did NOT have that shape</b> — they were flat to batch 8 and
        /// then dropped six-fold at batch 32, reproducibly, and nothing here explains why; it is recorded as
        /// unexplained rather than rationalised, and the shape that produced it no longer exists. See
        /// <c>ForwardCost_ByBatchSize</c> in
        /// <c>Tests/Diagnostics/LinearForwardParallelThresholdDiagnostics.cs</c>.</para>
        /// </summary>
        private static void ForwardColumnsParallel(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int batchSize,
            int inputSize,
            int outputSize)
        {
            // Pinned for the duration of the dispatch: OverfitParallel takes a static function pointer plus a
            // void* context, so the spans have to survive as raw pointers across the worker boundary. Same
            // shape as BackwardInput below.
            fixed (float* inputPtr = input, weightsPtr = weightsInputOutput, biasPtr = bias, outputPtr = output)
            {
                var context = new ForwardColumnContext(
                    inputPtr,
                    weightsPtr,
                    biasPtr,
                    outputPtr,
                    inputSize,
                    outputSize);

                // Grain 1, i.e. let the pool split the work evenly, and this was MEASURED rather than assumed.
                // Passing `Vector<float>.Count * 4` — so no worker gets less than one full unrolled block —
                // looked obviously right: at 32 workers a 1024-wide layer gives each worker half a block,
                // which cannot enter the kernel's 4-wide loop. It made no difference anywhere (1024x1024
                // 0.2252 -> 0.2278 ms, 4096x1024 0.2368 -> 0.2455, all inside the run-to-run spread), because
                // what actually decides the crossover is a fixed dispatch cost and not the unroll. Reverted
                // rather than kept, since an unmeasurable refinement is a claim somebody later has to
                // re-check. See `ForwardParallelWeightBytes`.
                OverfitParallel.For(0, batchSize * outputSize, &ForwardColumnRangeWorker, &context);
            }
        }

        /// <summary>
        /// One worker's slice of the flat <c>batchSize × outputSize</c> rectangle.
        ///
        /// <para><paramref name="rangeStart"/> and <paramref name="rangeEnd"/> index that rectangle in row-major
        /// order, so a slice generally begins part-way through one row and ends part-way through another. The
        /// loop walks it row by row, clamping each row's column range to what this worker owns — which is
        /// what keeps the split from having to align to row boundaries and therefore what lets a batch of 2
        /// still use every worker.</para>
        /// </summary>
        private static void ForwardColumnRangeWorker(
            int rangeStart,
            int rangeEnd,
            void* contextPtr)
        {
            ref readonly var context = ref Unsafe.AsRef<ForwardColumnContext>(contextPtr);
            var inputSize = context.InputSize;
            var outputSize = context.OutputSize;
            var weightCount = inputSize * outputSize;
            var position = rangeStart;

            // BOUND: each pass consumes at least one column and advances `position`, so at most
            // (rangeEnd - rangeStart) iterations.
            while (position < rangeEnd)
            {
                var row = position / outputSize;
                var rowStart = row * outputSize;
                var columnStart = position - rowStart;

                // The worker's range may end inside this row or run past it; take whichever comes first.
                var columnEnd = Math.Min(outputSize, rangeEnd - rowStart);

                ForwardInputMajorVector4(
                    new ReadOnlySpan<float>(context.Input + ((long)row * inputSize), inputSize),
                    new ReadOnlySpan<float>(context.Weights, weightCount),
                    new ReadOnlySpan<float>(context.Bias, outputSize),
                    new Span<float>(context.Output + ((long)rowStart), outputSize),
                    inputSize,
                    outputSize,
                    columnStart,
                    columnEnd);

                position = rowStart + columnEnd;
            }
        }

        private readonly struct ForwardColumnContext
        {
            public readonly float* Input;
            public readonly float* Weights;
            public readonly float* Bias;
            public readonly float* Output;
            public readonly int InputSize;
            public readonly int OutputSize;

            public ForwardColumnContext(
                float* input,
                float* weights,
                float* bias,
                float* output,
                int inputSize,
                int outputSize)
            {
                Input = input;
                Weights = weights;
                Bias = bias;
                Output = output;
                InputSize = inputSize;
                OutputSize = outputSize;
            }
        }

        /// <summary>Rows of the batch held in one register tile — see <see cref="ForwardOutputMajorTiled"/>.</summary>
        private const int OutputMajorRowTile = 4;

        /// <summary>Outputs held in one register tile, alongside <see cref="OutputMajorRowTile"/> rows.</summary>
        private const int OutputMajorColumnTile = 4;

        /// <summary>
        /// Batched narrow-output inference: a <see cref="OutputMajorRowTile"/>×<see cref="OutputMajorColumnTile"/>
        /// register tile over (batch rows × outputs), so each input element and each weight element is loaded
        /// once per tile instead of once per (row, output) pair.
        ///
        /// <para><b>Why this exists (`XC-80`).</b> The per-row path below calls <c>TensorPrimitives.Dot</c>
        /// once per output, and every one of those re-reads the whole input vector as well as its own weight
        /// row. On <c>Linear(784→10)</c> that is 15,680 FLOP against ~62 KB of L1 traffic — <b>0.25
        /// FLOP/byte</b> — which is why cost per row was FLAT from batch 1 to batch 256 while ONNX Runtime
        /// fell from 1,919 to 93 ns per row. The batch bought us nothing because nothing was being reused.</para>
        ///
        /// <para><b>Intensity is <c>MR·NR / (2(MR+NR))</c></b>: 1.00 FLOP/byte at 4×4 against 0.25 today. 4×8
        /// would give 1.33 but needs 32 vector accumulators — the entire AVX-512 register file — so it would
        /// spill. 4×4 needs 16, the same count the conv micro-kernel carries.</para>
        ///
        /// <para><b>Batch 1 never comes here</b>, and that is deliberate rather than incidental: a single row
        /// has nothing to tile against, and <c>Linear(784,10)</c> at batch 1 is this repository's published
        /// 237 ns / 8.3×-vs-ONNX-Runtime figure. It keeps the untouched path.</para>
        /// </summary>
        /// <summary>
        /// Whether the 512-bit tile is used instead of the portable <see cref="Vector{T}"/> one.
        ///
        /// <para><b>This exists because <see cref="Vector{T}"/> is NOT the machine's width.</b> Measured on
        /// this box: <c>Vector&lt;float&gt;.Count == 8</c> — 256 bits — while <c>Vector512.IsHardwareAccelerated</c>
        /// and AVX-512 support are both true. .NET caps <see cref="Vector{T}"/> at 256 bits unless
        /// <c>DOTNET_PreferredVectorBitWidth=512</c> is set, so every kernel written against it silently runs
        /// at half width on AVX-512 hardware. <c>Conv2DGemmKernels</c> has always sidestepped this with an
        /// explicit <see cref="Vector512{T}"/> path; this file did not, and ran at half width until
        /// 2026-08-17 (`XC-80`).</para>
        ///
        /// <para>The portable path is kept rather than replaced: it is what runs on anything without AVX-512,
        /// and it is the reference the 512 path is tested against.</para>
        /// </summary>
        internal static readonly bool UseAvx512Linear =
            CpuFeatures.HasAvx512
            && Environment.GetEnvironmentVariable(OverfitEnvironment.LinearAvx512) != "0";

        private static void ForwardOutputMajorTiled(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsOutputInput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int batchSize,
            int inputSize,
            int outputSize)
        {
            if (UseAvx512Linear)
            {
                ForwardOutputMajorTiled512(
                    input,
                    weightsOutputInput,
                    bias,
                    output,
                    batchSize,
                    inputSize,
                    outputSize);

                return;
            }

            var width = Vector<float>.Count;

            // Hoisted out of both loops: 64 bytes of stack reused by every tile. Allocating it per tile is a
            // stack overflow waiting for a large batch, and CA2014 says so.
            var sums = stackalloc float[OutputMajorRowTile * OutputMajorColumnTile];

            for (var b0 = 0; b0 < batchSize; b0 += OutputMajorRowTile)
            {
                var rows = Math.Min(OutputMajorRowTile, batchSize - b0);

                // Clamped to the last valid row/column so a partial tile still reads IN BOUNDS and the inner
                // loop stays branch-free; the padding lanes compute a duplicate result that is simply not
                // stored. Same device as MicroKernel8x32Avx512 — a separate edge kernel would be a second
                // implementation of the same contraction.
                var r0 = (b0 + Math.Min(0, rows - 1)) * inputSize;
                var r1 = (b0 + Math.Min(1, rows - 1)) * inputSize;
                var r2 = (b0 + Math.Min(2, rows - 1)) * inputSize;
                var r3 = (b0 + Math.Min(3, rows - 1)) * inputSize;

                for (var j0 = 0; j0 < outputSize; j0 += OutputMajorColumnTile)
                {
                    var columns = Math.Min(OutputMajorColumnTile, outputSize - j0);

                    var w0 = (j0 + Math.Min(0, columns - 1)) * inputSize;
                    var w1 = (j0 + Math.Min(1, columns - 1)) * inputSize;
                    var w2 = (j0 + Math.Min(2, columns - 1)) * inputSize;
                    var w3 = (j0 + Math.Min(3, columns - 1)) * inputSize;

                    // Sixteen accumulators as NAMED LOCALS, not an array: an array lives in memory and this
                    // is a register tile or it is nothing.
                    Vector<float> a00 = default, a01 = default, a02 = default, a03 = default;
                    Vector<float> a10 = default, a11 = default, a12 = default, a13 = default;
                    Vector<float> a20 = default, a21 = default, a22 = default, a23 = default;
                    Vector<float> a30 = default, a31 = default, a32 = default, a33 = default;

                    var i = 0;

                    // The contraction. Per step: four input vectors and four weight vectors feed sixteen
                    // FMAs — 1.00 FLOP/byte, against 0.25 for the dot-per-output path.
                    for (; i <= inputSize - width; i += width)
                    {
                        var x0 = new Vector<float>(input.Slice(r0 + i, width));
                        var x1 = new Vector<float>(input.Slice(r1 + i, width));
                        var x2 = new Vector<float>(input.Slice(r2 + i, width));
                        var x3 = new Vector<float>(input.Slice(r3 + i, width));

                        var v = new Vector<float>(weightsOutputInput.Slice(w0 + i, width));
                        a00 += x0 * v;
                        a10 += x1 * v;
                        a20 += x2 * v;
                        a30 += x3 * v;

                        v = new Vector<float>(weightsOutputInput.Slice(w1 + i, width));
                        a01 += x0 * v;
                        a11 += x1 * v;
                        a21 += x2 * v;
                        a31 += x3 * v;

                        v = new Vector<float>(weightsOutputInput.Slice(w2 + i, width));
                        a02 += x0 * v;
                        a12 += x1 * v;
                        a22 += x2 * v;
                        a32 += x3 * v;

                        v = new Vector<float>(weightsOutputInput.Slice(w3 + i, width));
                        a03 += x0 * v;
                        a13 += x1 * v;
                        a23 += x2 * v;
                        a33 += x3 * v;
                    }

                    // Horizontal reduction into the hoisted stack slots; the padding lanes of a partial
                    // tile are reduced too and then simply skipped below.
                    sums[0] = Vector.Sum(a00);
                    sums[1] = Vector.Sum(a01);
                    sums[2] = Vector.Sum(a02);
                    sums[3] = Vector.Sum(a03);
                    sums[4] = Vector.Sum(a10);
                    sums[5] = Vector.Sum(a11);
                    sums[6] = Vector.Sum(a12);
                    sums[7] = Vector.Sum(a13);
                    sums[8] = Vector.Sum(a20);
                    sums[9] = Vector.Sum(a21);
                    sums[10] = Vector.Sum(a22);
                    sums[11] = Vector.Sum(a23);
                    sums[12] = Vector.Sum(a30);
                    sums[13] = Vector.Sum(a31);
                    sums[14] = Vector.Sum(a32);
                    sums[15] = Vector.Sum(a33);

                    for (var r = 0; r < rows; r++)
                    {
                        for (var cIndex = 0; cIndex < columns; cIndex++)
                        {
                            var sum = sums[(r * OutputMajorColumnTile) + cIndex];

                            // Scalar tail, when inputSize is not a whole number of vectors.
                            for (var t = i; t < inputSize; t++)
                            {
                                sum += input[((b0 + r) * inputSize) + t]
                                       * weightsOutputInput[((j0 + cIndex) * inputSize) + t];
                            }

                            output[((b0 + r) * outputSize) + j0 + cIndex] = sum + bias[j0 + cIndex];
                        }
                    }
                }
            }
        }

        /// <summary>
        /// <see cref="ForwardOutputMajorTiled"/> at the machine's real width: sixteen lanes per vector and an
        /// explicit fused multiply-add, instead of eight lanes and a multiply the JIT may or may not fuse.
        ///
        /// <para>Structurally identical to the portable version on purpose — same 4×4 tile, same clamped
        /// bases for partial tiles, same scalar tail. Keeping the two in the same shape is what makes the
        /// parity test meaningful: it runs whichever one this machine selects, so a divergence between them
        /// shows up as a disagreement with the naive oracle rather than as two kernels nobody compares.</para>
        /// </summary>
        private static void ForwardOutputMajorTiled512(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsOutputInput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int batchSize,
            int inputSize,
            int outputSize)
        {
            const int Width = 16;

            var sums = stackalloc float[OutputMajorRowTile * OutputMajorColumnTile];

            fixed (float* inputBase = input,
                   weightsBase = weightsOutputInput,
                   biasBase = bias,
                   outputBase = output)
            {
                for (var b0 = 0; b0 < batchSize; b0 += OutputMajorRowTile)
                {
                    var rows = Math.Min(OutputMajorRowTile, batchSize - b0);

                    // Clamped to the last valid row so a partial tile reads in bounds; the duplicate lanes are
                    // computed and then not stored.
                    var x0 = inputBase + ((long)(b0 + Math.Min(0, rows - 1)) * inputSize);
                    var x1 = inputBase + ((long)(b0 + Math.Min(1, rows - 1)) * inputSize);
                    var x2 = inputBase + ((long)(b0 + Math.Min(2, rows - 1)) * inputSize);
                    var x3 = inputBase + ((long)(b0 + Math.Min(3, rows - 1)) * inputSize);

                    for (var j0 = 0; j0 < outputSize; j0 += OutputMajorColumnTile)
                    {
                        var columns = Math.Min(OutputMajorColumnTile, outputSize - j0);

                        var w0 = weightsBase + ((long)(j0 + Math.Min(0, columns - 1)) * inputSize);
                        var w1 = weightsBase + ((long)(j0 + Math.Min(1, columns - 1)) * inputSize);
                        var w2 = weightsBase + ((long)(j0 + Math.Min(2, columns - 1)) * inputSize);
                        var w3 = weightsBase + ((long)(j0 + Math.Min(3, columns - 1)) * inputSize);

                        Vector512<float> a00 = default, a01 = default, a02 = default, a03 = default;
                        Vector512<float> a10 = default, a11 = default, a12 = default, a13 = default;
                        Vector512<float> a20 = default, a21 = default, a22 = default, a23 = default;
                        Vector512<float> a30 = default, a31 = default, a32 = default, a33 = default;

                        var i = 0;

                        for (; i <= inputSize - Width; i += Width)
                        {
                            var r0 = Vector512.Load(x0 + i);
                            var r1 = Vector512.Load(x1 + i);
                            var r2 = Vector512.Load(x2 + i);
                            var r3 = Vector512.Load(x3 + i);

                            var v = Vector512.Load(w0 + i);
                            a00 = Avx512F.FusedMultiplyAdd(r0, v, a00);
                            a10 = Avx512F.FusedMultiplyAdd(r1, v, a10);
                            a20 = Avx512F.FusedMultiplyAdd(r2, v, a20);
                            a30 = Avx512F.FusedMultiplyAdd(r3, v, a30);

                            v = Vector512.Load(w1 + i);
                            a01 = Avx512F.FusedMultiplyAdd(r0, v, a01);
                            a11 = Avx512F.FusedMultiplyAdd(r1, v, a11);
                            a21 = Avx512F.FusedMultiplyAdd(r2, v, a21);
                            a31 = Avx512F.FusedMultiplyAdd(r3, v, a31);

                            v = Vector512.Load(w2 + i);
                            a02 = Avx512F.FusedMultiplyAdd(r0, v, a02);
                            a12 = Avx512F.FusedMultiplyAdd(r1, v, a12);
                            a22 = Avx512F.FusedMultiplyAdd(r2, v, a22);
                            a32 = Avx512F.FusedMultiplyAdd(r3, v, a32);

                            v = Vector512.Load(w3 + i);
                            a03 = Avx512F.FusedMultiplyAdd(r0, v, a03);
                            a13 = Avx512F.FusedMultiplyAdd(r1, v, a13);
                            a23 = Avx512F.FusedMultiplyAdd(r2, v, a23);
                            a33 = Avx512F.FusedMultiplyAdd(r3, v, a33);
                        }

                        sums[0] = Vector512.Sum(a00);
                        sums[1] = Vector512.Sum(a01);
                        sums[2] = Vector512.Sum(a02);
                        sums[3] = Vector512.Sum(a03);
                        sums[4] = Vector512.Sum(a10);
                        sums[5] = Vector512.Sum(a11);
                        sums[6] = Vector512.Sum(a12);
                        sums[7] = Vector512.Sum(a13);
                        sums[8] = Vector512.Sum(a20);
                        sums[9] = Vector512.Sum(a21);
                        sums[10] = Vector512.Sum(a22);
                        sums[11] = Vector512.Sum(a23);
                        sums[12] = Vector512.Sum(a30);
                        sums[13] = Vector512.Sum(a31);
                        sums[14] = Vector512.Sum(a32);
                        sums[15] = Vector512.Sum(a33);

                        for (var r = 0; r < rows; r++)
                        {
                            for (var cIndex = 0; cIndex < columns; cIndex++)
                            {
                                var sum = sums[(r * OutputMajorColumnTile) + cIndex];
                                var rowInput = inputBase + ((long)(b0 + r) * inputSize);
                                var weightRow = weightsBase + ((long)(j0 + cIndex) * inputSize);

                                for (var t = i; t < inputSize; t++)
                                {
                                    sum += rowInput[t] * weightRow[t];
                                }

                                outputBase[((long)(b0 + r) * outputSize) + j0 + cIndex] =
                                    sum + biasBase[j0 + cIndex];
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ForwardOutputMajorDot(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsOutputInput,
            ReadOnlySpan<float> bias,
            Span<float> output)
        {
            var inputSize = input.Length;
            var outputSize = output.Length;

            for (var j = 0; j < outputSize; j++)
            {
                var wRow = weightsOutputInput.Slice(j * inputSize, inputSize);
                output[j] = TensorPrimitives.Dot(input, wRow) + bias[j];
            }
        }

        /// <summary>
        /// Computes output columns <c>[columnStart, columnEnd)</c> of one inference row.
        ///
        /// <para><b>The range is what lets one implementation serve both the serial and the parallel path.</b>
        /// The serial caller passes the whole width and gets byte-identical behaviour to the version before
        /// `XC-79`; a worker passes its own slice. Splitting the maths into two copies instead would have left
        /// two kernels to keep in step, and the parity test could only ever have checked one of them.</para>
        ///
        /// <para><paramref name="outputSize"/> stays the FULL layer width even when the range is narrower —
        /// it is the row stride of the input-major weight matrix, not the amount of work. Passing the range
        /// width here instead would read the wrong weights and still produce plausible numbers.</para>
        /// </summary>
        private static void ForwardInputMajorVector4(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize,
            int columnStart,
            int columnEnd)
        {
            if (!Vector.IsHardwareAccelerated ||
                outputSize < Vector<float>.Count * 4)
            {
                ForwardInputMajorVector1(
                    input,
                    weightsInputOutput,
                    bias,
                    output,
                    inputSize,
                    outputSize);

                return;
            }

            if (UseAvx512Linear)
            {
                ForwardInputMajorVector4Avx512(
                    input,
                    weightsInputOutput,
                    bias,
                    output,
                    inputSize,
                    outputSize,
                    columnStart,
                    columnEnd);

                return;
            }

            var vectorWidth = Vector<float>.Count;
            var blockWidth = vectorWidth * 4;

            var j = columnStart;

            for (; j <= columnEnd - blockWidth; j += blockWidth)
            {
                var acc0 = new Vector<float>(bias.Slice(j, vectorWidth));
                var acc1 = new Vector<float>(bias.Slice(j + vectorWidth, vectorWidth));
                var acc2 = new Vector<float>(bias.Slice(j + vectorWidth * 2, vectorWidth));
                var acc3 = new Vector<float>(bias.Slice(j + vectorWidth * 3, vectorWidth));

                for (var i = 0; i < inputSize; i++)
                {
                    var x = new Vector<float>(input[i]);
                    var rowBase = i * outputSize + j;

                    acc0 += x * new Vector<float>(weightsInputOutput.Slice(rowBase, vectorWidth));
                    acc1 += x * new Vector<float>(weightsInputOutput.Slice(rowBase + vectorWidth, vectorWidth));
                    acc2 += x * new Vector<float>(weightsInputOutput.Slice(rowBase + vectorWidth * 2, vectorWidth));
                    acc3 += x * new Vector<float>(weightsInputOutput.Slice(rowBase + vectorWidth * 3, vectorWidth));
                }

                acc0.CopyTo(output.Slice(j, vectorWidth));
                acc1.CopyTo(output.Slice(j + vectorWidth, vectorWidth));
                acc2.CopyTo(output.Slice(j + vectorWidth * 2, vectorWidth));
                acc3.CopyTo(output.Slice(j + vectorWidth * 3, vectorWidth));
            }

            for (; j <= columnEnd - vectorWidth; j += vectorWidth)
            {
                var acc = new Vector<float>(bias.Slice(j, vectorWidth));

                for (var i = 0; i < inputSize; i++)
                {
                    acc += new Vector<float>(input[i]) *
                           new Vector<float>(weightsInputOutput.Slice(i * outputSize + j, vectorWidth));
                }

                acc.CopyTo(output.Slice(j, vectorWidth));
            }

            for (; j < columnEnd; j++)
            {
                var sum = bias[j];

                for (var i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weightsInputOutput[i * outputSize + j];
                }

                output[j] = sum;
            }
        }

        /// <summary>
        /// <see cref="ForwardInputMajorVector4"/> at 512 bits: four <see cref="Vector512{T}"/> accumulators
        /// spanning 64 output columns, fed by an explicit fused multiply-add.
        ///
        /// <para><b>What widening does and does not buy here.</b> This kernel streams the weight matrix — per
        /// input element it broadcasts one scalar and loads four vectors, so arithmetic intensity is about
        /// 0.5 FLOP/byte either way and stays bandwidth-bound. The gain is therefore not intensity but
        /// instruction count: half as many loop iterations and 128-byte loads instead of 64. **That is a
        /// weaker argument than the one behind the tiled kernel, so treat the number this produces as the
        /// evidence rather than the reasoning above** (`XC-81`).</para>
        ///
        /// <para>No <see cref="ForwardInputMajorVector1"/> fallback: an <paramref name="outputSize"/> between
        /// 32 and 63 simply skips the 64-column block loop and is served by the 16-wide and scalar loops,
        /// which is strictly better than dropping to a one-vector kernel.</para>
        /// </summary>
        private static void ForwardInputMajorVector4Avx512(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize,
            int columnStart,
            int columnEnd)
        {
            const int Width = 16;
            const int BlockWidth = Width * 4;

            fixed (float* inputBase = input,
                   weightsBase = weightsInputOutput,
                   biasBase = bias,
                   outputBase = output)
            {
                var j = columnStart;

                for (; j <= columnEnd - BlockWidth; j += BlockWidth)
                {
                    var acc0 = Vector512.Load(biasBase + j);
                    var acc1 = Vector512.Load(biasBase + j + Width);
                    var acc2 = Vector512.Load(biasBase + j + (Width * 2));
                    var acc3 = Vector512.Load(biasBase + j + (Width * 3));

                    for (var i = 0; i < inputSize; i++)
                    {
                        var x = Vector512.Create(inputBase[i]);
                        var rowBase = weightsBase + ((long)i * outputSize) + j;

                        acc0 = Avx512F.FusedMultiplyAdd(x, Vector512.Load(rowBase), acc0);
                        acc1 = Avx512F.FusedMultiplyAdd(x, Vector512.Load(rowBase + Width), acc1);
                        acc2 = Avx512F.FusedMultiplyAdd(x, Vector512.Load(rowBase + (Width * 2)), acc2);
                        acc3 = Avx512F.FusedMultiplyAdd(x, Vector512.Load(rowBase + (Width * 3)), acc3);
                    }

                    Vector512.Store(acc0, outputBase + j);
                    Vector512.Store(acc1, outputBase + j + Width);
                    Vector512.Store(acc2, outputBase + j + (Width * 2));
                    Vector512.Store(acc3, outputBase + j + (Width * 3));
                }

                for (; j <= columnEnd - Width; j += Width)
                {
                    var acc = Vector512.Load(biasBase + j);

                    for (var i = 0; i < inputSize; i++)
                    {
                        acc = Avx512F.FusedMultiplyAdd(
                            Vector512.Create(inputBase[i]),
                            Vector512.Load(weightsBase + ((long)i * outputSize) + j),
                            acc);
                    }

                    Vector512.Store(acc, outputBase + j);
                }

                for (; j < columnEnd; j++)
                {
                    var sum = biasBase[j];

                    for (var i = 0; i < inputSize; i++)
                    {
                        sum += inputBase[i] * weightsBase[((long)i * outputSize) + j];
                    }

                    outputBase[j] = sum;
                }
            }
        }

        private static void ForwardInputMajorVector1(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            if (!Vector.IsHardwareAccelerated ||
                outputSize < Vector<float>.Count)
            {
                ForwardInputMajorScalar(
                    input,
                    weightsInputOutput,
                    bias,
                    output,
                    inputSize,
                    outputSize);

                return;
            }

            var vectorWidth = Vector<float>.Count;
            var j = 0;

            for (; j <= outputSize - vectorWidth; j += vectorWidth)
            {
                var acc = new Vector<float>(bias.Slice(j, vectorWidth));

                for (var i = 0; i < inputSize; i++)
                {
                    var x = new Vector<float>(input[i]);
                    var w = new Vector<float>(weightsInputOutput.Slice(i * outputSize + j, vectorWidth));

                    acc += x * w;
                }

                acc.CopyTo(output.Slice(j, vectorWidth));
            }

            for (; j < outputSize; j++)
            {
                var sum = bias[j];

                for (var i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weightsInputOutput[i * outputSize + j];
                }

                output[j] = sum;
            }
        }

        private static void ForwardInputMajorScalar(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            bias.Slice(0, outputSize).CopyTo(output);

            for (var i = 0; i < inputSize; i++)
            {
                var x = input[i];
                var wBase = i * outputSize;

                for (var j = 0; j < outputSize; j++)
                {
                    output[j] += x * weightsInputOutput[wBase + j];
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Backward kernels — span-only, no AutogradNode dependency
        //
        // ParallelThreshold: 1_048_576 (1M ops). Much higher than TensorMath's 4096.
        // Rationale: Parallel.For has ~2-5 µs overhead + TPL allocation (~1-3 KB).
        // For small matrices (e.g., Linear(8,10) backward: 64×10×8 = 5120 ops)
        // sequential SIMD via TensorPrimitives.Dot is 10-50× faster than Parallel.For.
        // For large matrices (e.g., Linear(1352,64) backward: 5.5M ops) parallelism wins.
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Computes the input gradient:
        ///   gradInput[b, i] += sum_j(gradOutput[b, j] * weights[i, j])
        ///
        /// Equivalent to: gradInput += gradOutput @ weights^T
        /// Layout: weights[inputSize, outputSize] (input-major, same as training path).
        /// </summary>
        /// <summary>Parallel threshold for BackwardInput: 512K ops.</summary>
        internal const long BackwardInputThreshold = 524_288;

        public static void BackwardInput(
            ReadOnlySpan<float> gradOutput,    // [batchSize, outputSize]
            ReadOnlySpan<float> weights,        // [inputSize, outputSize]
            Span<float> gradInput,              // [batchSize, inputSize]  — accumulated in-place
            int batchSize,
            int inputSize,
            int outputSize)
        {
            // Threshold: 512K ops. Below threshold sequential SIMD is faster.
            if ((long)batchSize * inputSize * outputSize < BackwardInputThreshold)
            {
                for (var b = 0; b < batchSize; b++)
                {
                    BackwardInputRow(
                        gradOutput.Slice(b * outputSize, outputSize),
                        weights,
                        gradInput.Slice(b * inputSize, inputSize),
                        inputSize,
                        outputSize);
                }
                return;
            }

            // fixed pointer cannot be captured in Parallel.For lambda (CS1764).
            // OverfitParallel takes a static function pointer + void* context,
            // so closures aren't a concern — but we still want fixed/pinned spans
            // for the duration of the dispatch.
            fixed (float* goPtr = gradOutput, wPtr = weights, giPtr = gradInput)
            {
                var ctx = new BackwardInputContext(goPtr, wPtr, giPtr, inputSize, outputSize);
                OverfitParallel.For(0, batchSize, &BackwardInputChunkWorker, &ctx);
            }
        }

        private static void BackwardInputChunkWorker(
            int chunkStart,
            int chunkEnd,
            void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<BackwardInputContext>(contextPtr);
            for (var b = chunkStart; b < chunkEnd; b++)
            {
                BackwardInputWorker(b, ctx);
            }
        }

        private readonly struct BackwardInputContext
        {
            public readonly float* GradOutput;
            public readonly float* Weights;
            public readonly float* GradInput;
            public readonly int InputSize;
            public readonly int OutputSize;

            public BackwardInputContext(float* go, float* w, float* gi, int inputSize, int outputSize)
            {
                GradOutput = go;
                Weights = w;
                GradInput = gi;
                InputSize = inputSize;
                OutputSize = outputSize;
            }
        }

        private static void BackwardInputWorker(int b, BackwardInputContext ctx)
        {
            var goRow = new ReadOnlySpan<float>(ctx.GradOutput + b * ctx.OutputSize, ctx.OutputSize);
            var giRow = new Span<float>(ctx.GradInput + b * ctx.InputSize, ctx.InputSize);

            for (var i = 0; i < ctx.InputSize; i++)
            {
                giRow[i] += TensorPrimitives.Dot(
                    goRow,
                    new ReadOnlySpan<float>(ctx.Weights + i * ctx.OutputSize, ctx.OutputSize));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void BackwardInputRow(
            ReadOnlySpan<float> gradOutputRow,   // [outputSize]
            ReadOnlySpan<float> weights,          // [inputSize, outputSize]
            Span<float> gradInputRow,             // [inputSize]
            int inputSize,
            int outputSize)
        {
            // gradInput[i] += Dot(gradOutput, weights[i, :])
            // weights[i, :] = weights.Slice(i * outputSize, outputSize)
            for (var i = 0; i < inputSize; i++)
            {
                gradInputRow[i] += TensorPrimitives.Dot(
                    gradOutputRow,
                    weights.Slice(i * outputSize, outputSize));
            }
        }

        /// <summary>
        /// Accumulates weight gradients:
        ///   gradWeights[i, j] += sum_b(input[b, i] * gradOutput[b, j])
        ///
        /// Equivalent to: gradWeights += input^T @ gradOutput
        /// Layout: gradWeights[inputSize, outputSize] (input-major).
        /// </summary>
        /// <summary>Parallel threshold for AccumulateWeightGrad: 1M ops.</summary>
        internal const long AccumulateWeightGradThreshold = 1_048_576;

        public static void AccumulateWeightGrad(
            ReadOnlySpan<float> input,         // [batchSize, inputSize]
            ReadOnlySpan<float> gradOutput,    // [batchSize, outputSize]
            Span<float> gradWeights,           // [inputSize, outputSize]  — accumulated in-place
            int batchSize,
            int inputSize,
            int outputSize)
        {
            // Sequential path for small matrices — no overhead.
            if ((long)batchSize * inputSize * outputSize < AccumulateWeightGradThreshold)
            {
                AccumulateWeightGradSeq(input, gradOutput, gradWeights, batchSize, inputSize, outputSize);
                return;
            }

            // OverfitParallel: static function pointer + void* context, zero
            // allocation per dispatch. Pin spans for the duration of For().
            fixed (float* inPtr = input, goPtr = gradOutput, gwPtr = gradWeights)
            {
                var ctx = new AccumulateWeightGradContext(inPtr, goPtr, gwPtr, batchSize, inputSize, outputSize);
                OverfitParallel.For(0, inputSize, &AccumulateWeightGradChunkWorker, &ctx);
            }
        }

        private static void AccumulateWeightGradChunkWorker(
            int chunkStart,
            int chunkEnd,
            void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<AccumulateWeightGradContext>(contextPtr);
            for (var i = chunkStart; i < chunkEnd; i++)
            {
                AccumulateWeightGradWorker(i, ctx);
            }
        }

        private readonly struct AccumulateWeightGradContext
        {
            public readonly float* Input;
            public readonly float* GradOutput;
            public readonly float* GradWeights;
            public readonly int BatchSize;
            public readonly int InputSize;
            public readonly int OutputSize;

            public AccumulateWeightGradContext(float* inp, float* go, float* gw, int b, int n, int m)
            {
                Input = inp;
                GradOutput = go;
                GradWeights = gw;
                BatchSize = b;
                InputSize = n;
                OutputSize = m;
            }
        }

        private static void AccumulateWeightGradWorker(int i, AccumulateWeightGradContext ctx)
        {
            var gwRow = new Span<float>(ctx.GradWeights + i * ctx.OutputSize, ctx.OutputSize);

            for (var b = 0; b < ctx.BatchSize; b++)
            {
                var xi = ctx.Input[b * ctx.InputSize + i];
                var goRow = new ReadOnlySpan<float>(ctx.GradOutput + b * ctx.OutputSize, ctx.OutputSize);
                TensorPrimitives.MultiplyAdd(goRow, xi, gwRow, gwRow);
            }
        }

        private static void AccumulateWeightGradSeq(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> gradOutput,
            Span<float> gradWeights,
            int batchSize,
            int inputSize,
            int outputSize)
        {
            for (var b = 0; b < batchSize; b++)
            {
                var inRow = input.Slice(b * inputSize, inputSize);
                var gO = gradOutput.Slice(b * outputSize, outputSize);

                for (var i = 0; i < inputSize; i++)
                {
                    var xi = inRow[i];
                    var wRow = gradWeights.Slice(i * outputSize, outputSize);
                    TensorPrimitives.MultiplyAdd(gO, xi, wRow, wRow);
                }
            }
        }

        /// <summary>
        /// Accumulates bias gradients:
        ///   gradBias[j] += sum_b(gradOutput[b, j])
        ///
        /// Always sequential — bias is small (outputSize elements).
        /// </summary>
        public static void AccumulateBiasGrad(
            ReadOnlySpan<float> gradOutput,    // [batchSize, outputSize]
            Span<float> gradBias,              // [outputSize]  — accumulated in-place
            int batchSize,
            int outputSize)
        {
            for (var b = 0; b < batchSize; b++)
            {
                TensorPrimitives.Add(
                    gradOutput.Slice(b * outputSize, outputSize),
                    gradBias,
                    gradBias);
            }
        }


        // ─────────────────────────────────────────────────────────────────────
        // ForwardBatched — weight-stationary outer-product, no zero-skipping
        //
        // Layout: weights[inputSize, outputSize] (input-major).
        //
        // Algorithm: for each input feature k, broadcast W[k,:] across all batch
        // rows simultaneously. W[k,:] stays in L1 cache while all B rows are
        // processed. Eliminates zero-skipping branch mispredictions (post-ReLU
        // activations are ~50% zero → ~50% misprediction rate on the skip branch).
        //
        // Routing in TensorMath.Linear:
        //   batchSize * inputSize * outputSize < ForwardBatchedThreshold
        //     → ForwardBatched sequential (weight-stationary)
        //   otherwise
        //     → ForwardBatched + Parallel.For over K-chunks (TODO) or old path
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Threshold above which <see cref="ForwardBatched"/> switches to the
        /// parallel path in <see cref="DevOnBike.Overfit.Ops.TensorMath"/>.
        /// 500_000 chosen so that:
        ///   - Linear(64,10)   batch=64: 64*64*10 = 40K → sequential ✓
        ///   - Linear(1352,64) batch=64: 64*1352*64 = 5.5M → parallel ✓
        /// </summary>
        internal const long ForwardBatchedThreshold = 500_000;

        /// <summary>
        /// Batched forward pass without zero-skipping.
        /// Caller is responsible for initialising <paramref name="output"/> with
        /// the bias before calling (use <see cref="InitWithBias"/>).
        /// </summary>
        public static void ForwardBatched(
            ReadOnlySpan<float> input,    // [batchSize, inputSize]
            ReadOnlySpan<float> weights,  // [inputSize, outputSize] input-major
            Span<float> output,           // [batchSize, outputSize]  — bias pre-filled
            int batchSize,
            int inputSize,
            int outputSize)
        {
            // Weight-stationary outer product:
            //   for each input feature k:
            //     for each batch row b:
            //       output[b,:] += input[b,k] * W[k,:]
            //
            // W[k,:] (outputSize floats) is loaded once per k and reused for all B rows.
            // For outputSize=64 (256 bytes) this fits in L1 cache across all B=64 iterations.
            for (var k = 0; k < inputSize; k++)
            {
                var wRow = weights.Slice(k * outputSize, outputSize);

                for (var b = 0; b < batchSize; b++)
                {
                    var xi = input[b * inputSize + k];
                    TensorPrimitives.MultiplyAdd(
                        wRow,
                        xi,
                        output.Slice(b * outputSize, outputSize),
                        output.Slice(b * outputSize, outputSize));
                }
            }
        }

        /// <summary>
        /// Initialises the output buffer with bias, broadcasted across batchSize rows.
        /// Call before <see cref="ForwardBatched"/>.
        /// </summary>
        public static void InitWithBias(
            Span<float> output,
            ReadOnlySpan<float> bias,
            int batchSize,
            int outputSize)
        {
            for (var b = 0; b < batchSize; b++)
            {
                bias.CopyTo(output.Slice(b * outputSize, outputSize));
            }
        }

    }
}