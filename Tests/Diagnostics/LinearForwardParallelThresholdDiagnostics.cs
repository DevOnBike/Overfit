// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Kernels;

namespace DevOnBike.Overfit.Tests.Diagnostics
{
    /// <summary>
    /// Sweeps <see cref="LinearKernels.Forward"/> across layer sizes so
    /// <see cref="LinearKernels.ForwardParallelWeightBytes"/> is a measured crossover rather than a guess
    /// (`XC-79`).
    ///
    /// <para><b>How to read it, and why it takes two runs.</b> The threshold is a <c>const</c>, so a single
    /// process cannot exercise both sides of it. Run this once as shipped, then once with the constant
    /// temporarily set to <c>0</c> (everything parallel) — the crossover is the smallest weight size at which
    /// the all-parallel arm is faster. Below it the dispatch costs more than the extra cores return, and the
    /// published <c>Linear(784,10)</c> path lives well below it.</para>
    ///
    /// <para><b>The unit on the x-axis is bytes of weights, not FLOPs.</b> At batch 1 this kernel does half a
    /// flop per byte read, so it is bandwidth-bound and what decides whether more cores help is the size of
    /// the weight read. A FLOP-based sweep would put layers with the same arithmetic and very different
    /// memory traffic in the same bucket.</para>
    /// </summary>
    public sealed class LinearForwardParallelThresholdDiagnostics
    {
        private const int Repeats = 30;

        private readonly ITestOutputHelper _out;

        public LinearForwardParallelThresholdDiagnostics(ITestOutputHelper output) => _out = output;

        [LongFact("6s")]
        public void ForwardCost_ByWeightBytes()
        {
            (int Input, int Output)[] shapes =
            [
                (784, 10),      // the published 8.3x-vs-ORT layer — must stay serial
                (256, 128),
                (512, 256),
                (1024, 512),
                (1024, 1024),
                (2048, 1024),
                (4096, 1024),
                (25088, 4096),  // VGG-16 FC1, the layer XC-79 was filed on
            ];

            _out.WriteLine(
                $"threshold as built: {LinearKernels.ForwardParallelWeightBytes:N0} B   "
                + $"workers: {DevOnBike.Overfit.Runtime.OverfitParallel.WorkerCount}");
            _out.WriteLine($"{"layer",-16}{"weight KB",12}{"ms",10}{"GB/s",10}  path");

            foreach (var (inputSize, outputSize) in shapes)
            {
                var weightBytes = (long)inputSize * outputSize * sizeof(float);
                var parallel = outputSize >= 32 && weightBytes >= LinearKernels.ForwardParallelWeightBytes;

                var input = Filled(inputSize);
                var weightsInputOutput = Filled(inputSize * outputSize);
                var weightsOutputInput = new float[inputSize * outputSize];
                var bias = Filled(outputSize);
                var output = new float[outputSize];

                LinearKernels.TransposeInputOutputToOutputInput(
                    weightsInputOutput, weightsOutputInput, inputSize, outputSize);

                // Warm-up outside the timed region: JIT, and the first touch of a large weight array.
                for (var i = 0; i < 3; i++)
                {
                    LinearKernels.Forward(
                        input, weightsInputOutput, weightsOutputInput, bias, output, inputSize, outputSize);
                }

                // Best-of-N rather than a mean: the minimum is the run least disturbed by whatever else the
                // machine did, and this is a latency question.
                var best = double.MaxValue;

                for (var run = 0; run < Repeats; run++)
                {
                    var started = ValueStopwatch.StartNew();

                    LinearKernels.Forward(
                        input, weightsInputOutput, weightsOutputInput, bias, output, inputSize, outputSize);

                    var elapsed = started.GetElapsedTime().TotalMilliseconds;

                    if (elapsed < best)
                    {
                        best = elapsed;
                    }
                }

                _out.WriteLine(
                    $"{inputSize + "x" + outputSize,-16}{weightBytes / 1024.0,12:F1}{best,10:F4}"
                    + $"{weightBytes / 1e6 / best,10:F1}  {(parallel ? "parallel" : "serial")}");
            }
        }

        /// <summary>
        /// The same kernel across batch sizes, which is a different question from the sweep above.
        ///
        /// <para><b>Why it needs its own measurement.</b> The threshold decides whether to parallelise; the
        /// batch decides how many times that decision is acted on. A shape that dispatches once per row pays
        /// the fixed dispatch cost <c>batchSize</c> times, so it can be optimal at batch 1 and badly wrong at
        /// batch 64 — and the per-row time is the number that shows it, because a correct implementation
        /// keeps it roughly flat as the batch grows.</para>
        /// </summary>
        [LongFact("8s")]
        public void ForwardCost_ByBatchSize()
        {
            const int inputSize = 2048;
            const int outputSize = 1024;

            var weightBytes = (long)inputSize * outputSize * sizeof(float);

            _out.WriteLine(
                $"layer {inputSize}x{outputSize} = {weightBytes / 1024.0:N0} KB of weights   "
                + $"threshold {LinearKernels.ForwardParallelWeightBytes:N0} B   "
                + $"workers {DevOnBike.Overfit.Runtime.OverfitParallel.WorkerCount}");
            _out.WriteLine($"{"batch",8}{"total ms",12}{"ms/row",12}{"GB/s",10}");

            var weightsInputOutput = Filled(inputSize * outputSize);
            var weightsOutputInput = new float[inputSize * outputSize];
            var bias = Filled(outputSize);

            LinearKernels.TransposeInputOutputToOutputInput(
                weightsInputOutput, weightsOutputInput, inputSize, outputSize);

            foreach (var batchSize in new[] { 1, 2, 4, 8, 32, 64 })
            {
                var input = Filled(batchSize * inputSize);
                var output = new float[batchSize * outputSize];

                for (var i = 0; i < 3; i++)
                {
                    LinearKernels.Forward(
                        input, weightsInputOutput, weightsOutputInput, bias, output, inputSize, outputSize);
                }

                var best = double.MaxValue;

                for (var run = 0; run < Repeats; run++)
                {
                    var started = ValueStopwatch.StartNew();

                    LinearKernels.Forward(
                        input, weightsInputOutput, weightsOutputInput, bias, output, inputSize, outputSize);

                    var elapsed = started.GetElapsedTime().TotalMilliseconds;

                    if (elapsed < best)
                    {
                        best = elapsed;
                    }
                }

                _out.WriteLine(
                    $"{batchSize,8}{best,12:F4}{best / batchSize,12:F4}"
                    + $"{weightBytes * batchSize / 1e6 / best,10:F1}");
            }
        }

        private static float[] Filled(int length)
        {
            var values = new float[length];
            var state = 0x9E3779B9u;

            for (var i = 0; i < length; i++)
            {
                state = (state * 1664525u) + 1013904223u;
                values[i] = (((state & 0x00FFFFFF) / 16777216f) * 2f) - 1f;
            }

            return values;
        }
    }
}
