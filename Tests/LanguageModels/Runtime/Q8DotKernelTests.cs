// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Correctness + F32-parity tests for <see cref="Q8DotKernel"/> — the INT8
    /// quantized dot product (step 2.1 of the decode-kernel plan).
    /// </summary>
    public sealed class Q8DotKernelTests
    {
        private static (sbyte[] quants, float[] scales) Quantize(float[] values)
        {
            var quants = new sbyte[values.Length];
            var scales = new float[values.Length / Q8DotKernel.BlockSize];
            Q8DotKernel.Quantize(values, quants, scales);
            return (quants, scales);
        }

        private static double F32Dot(float[] a, float[] b)
        {
            var sum = 0.0;
            for (var i = 0; i < a.Length; i++)
            {
                sum += (double)a[i] * b[i];
            }
            return sum;
        }

        [Theory]
        [InlineData(32, 1)]
        [InlineData(320, 2)]
        [InlineData(2048, 3)]
        [InlineData(11008, 4)]
        public void Dot_MatchesF32_WithinQuantizationNoise(int length, int seed)
        {
            var rng = new Random(seed);
            var a = new float[length];
            var b = new float[length];
            for (var i = 0; i < length; i++)
            {
                a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                b[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            var reference = F32Dot(a, b);

            var (aq, aScales) = Quantize(a);
            var (bq, bScales) = Quantize(b);
            var q8 = Q8DotKernel.Dot(aq, aScales, bq, bScales, length);

            var absErr = Math.Abs(q8 - reference);
            var relErr = absErr / Math.Max(1e-6, Math.Abs(reference));

            // Q8_0 carries ~8 bits/value; the dot stays within ~1-2% relative.
            // Combined tolerance: relative OR a small absolute, so a reference
            // that happens to land near zero does not blow the ratio up.
            Assert.True(
                relErr < 0.03 || absErr < 0.5,
                $"length={length} q8={q8} ref={reference} absErr={absErr} relErr={relErr:F4}");
        }

        [Fact]
        public void Dot_AllZeros_IsZero()
        {
            const int length = 256;
            var (aq, aScales) = Quantize(new float[length]);
            var (bq, bScales) = Quantize(new float[length]);

            Assert.Equal(0f, Q8DotKernel.Dot(aq, aScales, bq, bScales, length));
        }

        [Fact]
        public void Dot_AllOnes_EqualsLength()
        {
            const int length = 64;
            var a = new float[length];
            var b = new float[length];
            Array.Fill(a, 1f);
            Array.Fill(b, 1f);

            var (aq, aScales) = Quantize(a);
            var (bq, bScales) = Quantize(b);

            // Constant block → scale = 1/127, quant = 127 → reconstructs to 1.0
            // exactly; the dot is exact at length.
            Assert.Equal(length, Q8DotKernel.Dot(aq, aScales, bq, bScales, length), 3);
        }

        [Fact]
        public void Quantize_RoundTrip_WithinHalfAStep()
        {
            const int length = 512;
            var rng = new Random(99);
            var values = new float[length];
            for (var i = 0; i < length; i++)
            {
                values[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
            }

            var (quants, scales) = Quantize(values);

            for (var i = 0; i < length; i++)
            {
                var scale = scales[i / Q8DotKernel.BlockSize];
                var reconstructed = scale * quants[i];
                // Quantization error is bounded by half a step (scale / 2).
                Assert.True(
                    Math.Abs(reconstructed - values[i]) <= scale * 0.5f + 1e-6f,
                    $"i={i} value={values[i]} reconstructed={reconstructed} scale={scale}");
            }
        }

        [Fact]
        public void Quantize_RejectsNonBlockMultipleLength()
        {
            Assert.Throws<ArgumentException>(() =>
                Q8DotKernel.Quantize(new float[33], new sbyte[33], new float[2]));
        }

        [Theory]
        [InlineData(2048, 64, false)]
        [InlineData(2048, 512, true)]
        [InlineData(128, 2048, false)]
        [InlineData(11008, 256, true)]
        public void Project_MatchesF32_WithinQuantizationNoise(int inputSize, int outputSize, bool withBias)
        {
            var rng = new Random(inputSize * 31 + outputSize);

            var input = new float[inputSize];
            for (var i = 0; i < inputSize; i++)
            {
                input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            // Output-major F32 weight: row o = the contraction vector for output o.
            var weightF32 = new float[outputSize * inputSize];
            for (var k = 0; k < weightF32.Length; k++)
            {
                weightF32[k] = (float)(rng.NextDouble() * 0.2 - 0.1);
            }

            var bias = withBias ? new float[outputSize] : [];
            for (var o = 0; o < bias.Length; o++)
            {
                bias[o] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            // Quantize the weight, one output row at a time.
            var blocksPerRow = inputSize / Q8DotKernel.BlockSize;
            var weightQuants = new sbyte[outputSize * inputSize];
            var weightScales = new float[outputSize * blocksPerRow];
            for (var o = 0; o < outputSize; o++)
            {
                Q8DotKernel.Quantize(
                    weightF32.AsSpan(o * inputSize, inputSize),
                    weightQuants.AsSpan(o * inputSize, inputSize),
                    weightScales.AsSpan(o * blocksPerRow, blocksPerRow));
            }

            // F32 reference.
            var reference = new float[outputSize];
            for (var o = 0; o < outputSize; o++)
            {
                var sum = 0.0;
                for (var i = 0; i < inputSize; i++)
                {
                    sum += (double)input[i] * weightF32[o * inputSize + i];
                }
                reference[o] = (float)((withBias ? bias[o] : 0f) + sum);
            }

            // Q8 projection.
            var outQ8 = new float[outputSize];
            var inputQuants = new sbyte[inputSize];
            var inputScales = new float[blocksPerRow];
            Q8DotKernel.Project(
                input, weightQuants, weightScales, bias, outQ8,
                inputSize, outputSize, inputQuants, inputScales);

            // L2 relative error — the robust accuracy metric (no per-output
            // blow-up when an individual reference lands near zero).
            double errorSq = 0.0;
            double referenceSq = 0.0;
            for (var o = 0; o < outputSize; o++)
            {
                var d = (double)outQ8[o] - reference[o];
                errorSq += d * d;
                referenceSq += (double)reference[o] * reference[o];
            }

            var l2Rel = Math.Sqrt(errorSq / Math.Max(1e-12, referenceSq));
            Assert.True(
                l2Rel < 0.03,
                $"L2 relative error {l2Rel:F4} (inputSize={inputSize}, outputSize={outputSize}, bias={withBias})");
        }

        [Fact]
        public void ProjectParallel_IsBitIdenticalToSequentialProject()
        {
            const int inputSize = 2048;
            const int outputSize = 4096;
            var rng = new Random(7);

            var input = new float[inputSize];
            for (var i = 0; i < inputSize; i++)
            {
                input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            var weightF32 = new float[outputSize * inputSize];
            for (var k = 0; k < weightF32.Length; k++)
            {
                weightF32[k] = (float)(rng.NextDouble() * 0.2 - 0.1);
            }

            var weight = Q8Weight.QuantizeRows(weightF32, outputSize, inputSize);
            var blocksPerRow = inputSize / Q8DotKernel.BlockSize;

            // Sequential reference.
            var sequential = new float[outputSize];
            Q8DotKernel.Project(
                input, weight.Quants, weight.Scales, [], sequential,
                inputSize, outputSize, new sbyte[inputSize], new float[blocksPerRow]);

            // Parallel — each output is computed independently, so it must be
            // bit-identical to the sequential projection.
            var parallel = new float[outputSize];
            Q8DotKernel.ProjectParallel(
                input, weight, [], parallel, new sbyte[inputSize], new float[blocksPerRow]);

            for (var o = 0; o < outputSize; o++)
            {
                Assert.Equal(sequential[o], parallel[o]);
            }
        }
    }
}
