// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// Is <see cref="SingleTokenProjectionKernel.ParallelWorkThreshold"/> (1,000,000 elements) right on a
    /// many-core desktop? Measured on a phone it is far too high — SmolLM2-135M's 576x1536 = 884,736 FFN
    /// matmuls fell 12% below it, ran sequentially, and cost 46% of decode wall time (see
    /// <c>docs/measured-baselines.md</c>). Lowering it there gave +16%. This benchmark asks the same
    /// question on the 32-core box, because the constant is global and one machine's answer is not the
    /// other's.
    ///
    /// <para>
    /// <b>Why not a single-matrix microbenchmark.</b> One 576x1536 matrix is 3.5 MB and lives in L3 on this
    /// box, so a loop over it measures a cache-resident matmul — the opposite of decode, which streams every
    /// weight in the model from DRAM exactly once per token. That benchmark would answer confidently and
    /// wrongly. This one allocates <see cref="MatrixCount"/> DISTINCT weight matrices totalling well past
    /// L3 and walks them once per iteration, which is decode's memory regime: no reuse, bandwidth-bound.
    /// </para>
    ///
    /// <para>
    /// <b>One lever.</b> Both arms run the identical chain over the identical data in the identical process;
    /// the only difference is which path <see cref="SingleTokenProjectionKernel.ProjectParallel"/> takes,
    /// selected through <see cref="SingleTokenProjectionKernel.ParallelWorkThresholdOverride"/>. Comparing
    /// two builds instead would fold the machine's drift into the result.
    /// </para>
    ///
    /// <para>
    /// <b>Canary.</b> <see cref="Sequential_Canary"/> is a second copy of the sequential arm. It is
    /// untouched by the lever, so if the two sequential arms disagree by more than their error the box
    /// moved during the run and the whole table is void — read it before reading the ratio.
    /// </para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*DecodeProjectionThreshold*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class DecodeProjectionThresholdBenchmark
    {
        // 512 MB of weights at 4 bytes/float, sized past this box's L3 so nothing is reused across matrices.
        private const long TargetWeightBytes = 512L * 1024 * 1024;

        private float[][] _weights = null!;
        private float[] _input = null!;
        private float[] _output = null!;
        private float _checksum;

        /// <summary>Rows per matrix — the model's hidden size. 576 is SmolLM2-135M's.</summary>
        [Params(576)]
        public int InputSize
        {
            get; set;
        }

        /// <summary>
        /// Columns per matrix. 576 is an attention projection, 1536 the FFN (the shape that lands just
        /// below the constant), 4096 comfortably above it.
        /// </summary>
        [Params(576, 1536, 4096)]
        public int OutputSize
        {
            get; set;
        }

        public int MatrixCount
        {
            get; private set;
        }

        [GlobalSetup]
        public void Setup()
        {
            var elementsPerMatrix = (long)InputSize * OutputSize;
            MatrixCount = (int)Math.Max(1, TargetWeightBytes / (elementsPerMatrix * sizeof(float)));

            _weights = new float[MatrixCount][];
            var rng = new Random(4242);
            for (var m = 0; m < MatrixCount; m++)
            {
                var matrix = new float[elementsPerMatrix];
                for (var i = 0; i < matrix.Length; i++)
                {
                    matrix[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                }
                _weights[m] = matrix;
            }

            _input = new float[InputSize];
            for (var i = 0; i < _input.Length; i++)
            {
                _input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            _output = new float[OutputSize];
        }

        [Benchmark(Baseline = true)]
        public float Sequential()
        {
            return RunSequential();
        }

        [Benchmark]
        public float Sequential_Canary()
        {
            return RunSequential();
        }

        [Benchmark]
        public float Parallel()
        {
            // 1 = "always parallel", so this arm exercises the parallel path at every OutputSize, including
            // the ones the shipped constant would send down the sequential path.
            SingleTokenProjectionKernel.ParallelWorkThresholdOverride = 1;
            try
            {
                var checksum = 0f;
                for (var m = 0; m < _weights.Length; m++)
                {
                    SingleTokenProjectionKernel.ProjectParallel(
                        _input, _weights[m], ReadOnlySpan<float>.Empty, _output, InputSize, OutputSize);
                    checksum += _output[0];
                }
                _checksum = checksum;
                return checksum;
            }
            finally
            {
                SingleTokenProjectionKernel.ParallelWorkThresholdOverride = null;
            }
        }

        private float RunSequential()
        {
            var checksum = 0f;
            for (var m = 0; m < _weights.Length; m++)
            {
                SingleTokenProjectionKernel.Project(
                    _input, _weights[m], ReadOnlySpan<float>.Empty, _output, InputSize, OutputSize);
                checksum += _output[0];
            }
            _checksum = checksum;
            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            GC.KeepAlive(_checksum);
        }
    }
}
