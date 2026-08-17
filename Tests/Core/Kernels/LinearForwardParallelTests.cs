// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Kernels;

namespace DevOnBike.Overfit.Tests.Core.Kernels
{
    /// <summary>
    /// Forward inference of <see cref="LinearKernels"/> is correct on <b>both sides</b> of the threshold that
    /// decides whether the output columns are split across workers (`XC-79`).
    ///
    /// <para><b>The oracle is an independent naive triple loop, deliberately.</b> The obvious alternative —
    /// run the same kernel twice and compare — shares every bug the kernel has, and a column split is exactly
    /// the kind of change that produces self-consistent nonsense (each worker computing the right answer for
    /// the wrong columns, or overlapping ranges silently writing twice). A separate implementation of the
    /// definition cannot agree with a mis-indexed kernel by accident.</para>
    ///
    /// <para><b>What is NOT asserted, and why.</b> Bit-identity between the serial and the parallel path is
    /// not checked, because <c>OverfitParallel</c> resolves its worker count once at process start, so a
    /// single test process cannot exercise both. It is also not where the risk is: the split is over output
    /// columns, and each output column still sums over the inputs in the same order regardless of which
    /// worker owns it — the split cannot reorder a summation. Mis-indexing can, and that is what the oracle
    /// catches.</para>
    /// </summary>
    public sealed class LinearForwardParallelTests
    {
        [Theory]
        // Below the threshold: the serial vectorised path. 128 x 64 = 32 KB of weights.
        [InlineData(128, 64, 1)]
        // At the threshold: 2048 x 1024 x 4 B = exactly 8 MB, so the parallel path runs.
        [InlineData(2048, 1024, 1)]
        // Above it, with a width that is NOT a multiple of the 4-vector block, so a worker's range ends in
        // the remainder loops rather than on a clean boundary.
        [InlineData(2048, 1035, 1)]
        // Batched. The dispatch covers the flat batchSize x outputSize rectangle, so a worker's range
        // generally STARTS AND ENDS PART-WAY THROUGH A ROW — the case row-aligned splitting never produces
        // and the one most likely to write into the wrong row.
        [InlineData(1024, 512, 4)]
        [InlineData(1024, 512, 8)]
        // A batch that crosses the threshold only because of the batch: one row is 2 MB, eight rows are
        // 16 MB. Below, the same layer at batch 1 stays serial — so this pair covers both sides of the
        // batch-aware gate on one shape.
        [InlineData(512, 1024, 8)]
        [InlineData(512, 1024, 1)]
        // Batched with a non-block-aligned width: worker ranges land mid-row AND mid-block.
        [InlineData(1024, 517, 8)]
        public void Forward_MatchesANaiveReference_OnBothSidesOfTheParallelThreshold(
            int inputSize,
            int outputSize,
            int batchSize)
        {
            var input = Deterministic(inputSize * batchSize, seed: 11);
            var weightsInputOutput = Deterministic(inputSize * outputSize, seed: 22);
            var bias = Deterministic(outputSize, seed: 33);
            var weightsOutputInput = new float[inputSize * outputSize];

            LinearKernels.TransposeInputOutputToOutputInput(
                weightsInputOutput,
                weightsOutputInput,
                inputSize,
                outputSize);

            var actual = new float[outputSize * batchSize];

            LinearKernels.Forward(
                input,
                weightsInputOutput,
                weightsOutputInput,
                bias,
                actual,
                inputSize,
                outputSize);

            // Every row gets its OWN slice of the input, so a worker that writes the right values into the
            // wrong row is caught. A single shared input would make all rows identical and hide exactly that.
            for (var b = 0; b < batchSize; b++)
            {
                var expected = NaiveForward(
                    input.AsSpan(b * inputSize, inputSize),
                    weightsInputOutput,
                    bias,
                    inputSize,
                    outputSize);

                // Relative tolerance: the kernel accumulates in vector lanes and the oracle accumulates
                // serially, so the two differ in summation order. Over 2048 terms that is a rounding
                // difference, not a disagreement — a mis-indexed column is off by orders of magnitude.
                for (var j = 0; j < outputSize; j++)
                {
                    var scale = MathF.Max(1f, MathF.Abs(expected[j]));

                    Assert.True(
                        MathF.Abs(actual[(b * outputSize) + j] - expected[j]) / scale < 1e-4f,
                        $"row {b} of {batchSize}, column {j} of {outputSize}: "
                        + $"kernel {actual[(b * outputSize) + j]}, naive reference {expected[j]}");
                }
            }
        }

        /// <summary>
        /// A large layer is split across workers and every column is written exactly once.
        ///
        /// <para>Separate from the parity theory above because it fails differently: overlapping worker
        /// ranges still produce the right VALUE for the overlapped columns (they are written twice with the
        /// same result), while a gap leaves a column untouched. Pre-filling the output with a sentinel is what
        /// makes a gap visible at all — the parity test would report it as a wrong value, but only if the
        /// sentinel happened to differ, and zero-filled output plus a zero-ish expected value would hide
        /// it.</para>
        /// </summary>
        [Fact]
        public void Forward_WritesEveryColumn_WhenTheWorkIsSplitAcrossWorkers()
        {
            const int inputSize = 2048;
            const int outputSize = 1024;

            var input = Deterministic(inputSize, seed: 44);
            var weightsInputOutput = Deterministic(inputSize * outputSize, seed: 55);
            var bias = Deterministic(outputSize, seed: 66);
            var weightsOutputInput = new float[inputSize * outputSize];

            LinearKernels.TransposeInputOutputToOutputInput(
                weightsInputOutput,
                weightsOutputInput,
                inputSize,
                outputSize);

            const float sentinel = -987654f;
            var output = new float[outputSize];
            Array.Fill(output, sentinel);

            LinearKernels.Forward(
                input,
                weightsInputOutput,
                weightsOutputInput,
                bias,
                output,
                inputSize,
                outputSize);

            for (var j = 0; j < outputSize; j++)
            {
                Assert.NotEqual(sentinel, output[j]);
            }
        }

        /// <summary>
        /// The threshold is expressed in the unit the mechanism has — bytes of weights read — and is large
        /// enough that the repository's published <c>Linear(784, 10)</c> figure keeps the serial path.
        ///
        /// <para>This is a policy assertion, not a measurement: it pins the DECISION so that lowering the
        /// threshold to something that would dispatch a 237 ns layer fails here rather than in a benchmark
        /// nobody re-runs. `XC-79` records why that layer must not be dispatched.</para>
        /// </summary>
        [Fact]
        public void TheParallelThreshold_LeavesThePublishedSmallLayerOnTheSerialPath()
        {
            const long publishedLayerWeightBytes = 784L * 10 * sizeof(float);

            Assert.True(
                publishedLayerWeightBytes < LinearKernels.ForwardParallelWeightBytes,
                $"Linear(784,10) reads {publishedLayerWeightBytes} B of weights and the parallel threshold is "
                + $"{LinearKernels.ForwardParallelWeightBytes} B — it must stay below, or the published "
                + "~237 ns / 8.3x-vs-ONNX-Runtime number pays a worker dispatch it cannot afford.");
        }

        private static float[] NaiveForward(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            int inputSize,
            int outputSize)
        {
            var result = new float[outputSize];

            for (var j = 0; j < outputSize; j++)
            {
                var sum = bias[j];

                for (var i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weightsInputOutput[(i * outputSize) + j];
                }

                result[j] = sum;
            }

            return result;
        }

        private static float[] Deterministic(int length, int seed)
        {
            var values = new float[length];
            var state = (uint)(0x9E3779B9 + seed);

            for (var i = 0; i < length; i++)
            {
                state = (state * 1664525u) + 1013904223u;

                // Centred on zero so cancellation is exercised: an all-positive input would let a wrong
                // column pass a loose tolerance simply because every partial sum has the same sign.
                values[i] = (((state & 0x00FFFFFF) / 16777216f) * 2f) - 1f;
            }

            return values;
        }
    }
}
