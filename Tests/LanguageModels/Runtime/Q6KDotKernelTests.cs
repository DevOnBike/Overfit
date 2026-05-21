// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Synthetic correctness tests for <see cref="Q6KDotKernel"/> (step 3.3a —
    /// the Q6_K × Q8_K decode kernel, scalar core).
    ///
    /// Reference: the same Q6_K bytes decoded to F32 by the proven
    /// <see cref="GgmlDequant.DecodeQ6_KBlock"/>, then a plain F32 dot. The
    /// kernel's <c>Dot</c> is exact given the Q8_K activation, so the only
    /// divergence is the activation's 8-bit quantization — bounded here by an
    /// L2-relative tolerance.
    ///
    /// No model file needed — synthetic Q6_K blocks (sane FP16 d, random
    /// ql/qh/scales).
    /// </summary>
    [Trait("Category", "Quantization")]
    public sealed class Q6KDotKernelTests
    {
        private const int SuperBlockBytes = Q6KWeight.SuperBlockBytes;
        private const int SuperBlockElements = Q6KWeight.SuperBlockElements;

        [Fact]
        public void Project_MatchesF32Reference_WithinQ8ActivationNoise()
        {
            const int inputSize = 512;    // 2 super-blocks per row
            const int outputSize = 96;
            var rng = new Random(20260520);

            var blocks = BuildRandomBlocks(rng, outputSize, inputSize);
            var input = RandomVector(rng, inputSize);

            var weight = new Q6KWeight(blocks, inputSize, outputSize);
            var output = new float[outputSize];
            var superBlocksPerRow = inputSize / SuperBlockElements;
            var actQuants = new sbyte[inputSize];
            var actScales = new float[superBlocksPerRow];
            var actBsums = new short[superBlocksPerRow * Q6KDotKernel.GroupsPerSuperBlock];

            Q6KDotKernel.Project(input, weight, [], output, actQuants, actScales, actBsums);

            var reference = ReferenceProject(blocks, inputSize, outputSize, input);

            // L2-relative error: the weight side is exact, only the activation
            // is Q8-quantized — comparable to Q4_K's noise floor.
            double errSq = 0, refSq = 0;
            for (var o = 0; o < outputSize; o++)
            {
                var diff = (double)output[o] - reference[o];
                errSq += diff * diff;
                refSq += (double)reference[o] * reference[o];
            }
            var l2Relative = Math.Sqrt(errSq / refSq);

            Assert.True(l2Relative < 0.03,
                $"L2-relative error {l2Relative:F5} exceeds 3% — kernel diverges from the F32 decode.");
        }

        [Fact]
        public void ProjectParallel_IsBitIdenticalToProject()
        {
            const int inputSize = 512;
            const int outputSize = 96;
            var rng = new Random(424242);

            var blocks = BuildRandomBlocks(rng, outputSize, inputSize);
            var input = RandomVector(rng, inputSize);
            var bias = RandomVector(rng, outputSize);
            var weight = new Q6KWeight(blocks, inputSize, outputSize);
            var superBlocksPerRow = inputSize / SuperBlockElements;

            var sequential = new float[outputSize];
            var parallel = new float[outputSize];
            var actQuants = new sbyte[inputSize];
            var actScales = new float[superBlocksPerRow];
            var actBsums = new short[superBlocksPerRow * Q6KDotKernel.GroupsPerSuperBlock];

            Q6KDotKernel.Project(input, weight, bias, sequential, actQuants, actScales, actBsums);
            Q6KDotKernel.ProjectParallel(input, weight, bias, parallel, actQuants, actScales, actBsums);

            // Each output is an independent Dot — parallel must match exactly.
            for (var o = 0; o < outputSize; o++)
            {
                Assert.Equal(sequential[o], parallel[o]);
            }
        }

        [Fact]
        public void Project_AppliesBias()
        {
            const int inputSize = 256;
            const int outputSize = 32;
            var rng = new Random(99);

            var blocks = BuildRandomBlocks(rng, outputSize, inputSize);
            var input = RandomVector(rng, inputSize);
            var weight = new Q6KWeight(blocks, inputSize, outputSize);

            var actQuants = new sbyte[inputSize];
            var actScales = new float[1];
            var actBsums = new short[Q6KDotKernel.GroupsPerSuperBlock];

            var noBias = new float[outputSize];
            Q6KDotKernel.Project(input, weight, [], noBias, actQuants, actScales, actBsums);

            var bias = RandomVector(rng, outputSize);
            var withBias = new float[outputSize];
            Q6KDotKernel.Project(input, weight, bias, withBias, actQuants, actScales, actBsums);

            for (var o = 0; o < outputSize; o++)
            {
                Assert.Equal(noBias[o] + bias[o], withBias[o], 3);
            }
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        /// <summary>
        /// A synthetic Q6_K super-block: sane small positive FP16 d (random
        /// bits there could decode to NaN/Inf) over fully random ql / qh /
        /// scales — any such bytes are a valid Q6_K block.
        /// </summary>
        private static void WriteRandomBlock(Random rng, Span<byte> block210)
        {
            for (var i = 0; i < block210.Length; i++)
            {
                block210[i] = (byte)rng.Next(256);
            }

            // Overwrite d (at offset 208) with a sane small positive FP16.
            var d = (Half)(rng.NextDouble() * 0.05 + 0.001);
            BinaryPrimitives.WriteUInt16LittleEndian(block210.Slice(208, 2), BitConverter.HalfToUInt16Bits(d));
        }

        private static byte[] BuildRandomBlocks(Random rng, int outputSize, int inputSize)
        {
            var superBlocksPerRow = inputSize / SuperBlockElements;
            var blocks = new byte[outputSize * superBlocksPerRow * SuperBlockBytes];
            for (var b = 0; b < outputSize * superBlocksPerRow; b++)
            {
                WriteRandomBlock(rng, blocks.AsSpan(b * SuperBlockBytes, SuperBlockBytes));
            }
            return blocks;
        }

        private static float[] RandomVector(Random rng, int n)
        {
            var v = new float[n];
            for (var i = 0; i < n; i++)
            {
                v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
            return v;
        }

        /// <summary>F32 reference: decode each row's Q6_K blocks, plain dot with the input.</summary>
        private static float[] ReferenceProject(byte[] blocks, int inputSize, int outputSize, float[] input)
        {
            var superBlocksPerRow = inputSize / SuperBlockElements;
            var output = new float[outputSize];
            var row = new float[inputSize];

            for (var o = 0; o < outputSize; o++)
            {
                for (var sb = 0; sb < superBlocksPerRow; sb++)
                {
                    var blockIndex = o * superBlocksPerRow + sb;
                    GgmlDequant.DecodeQ6_KBlock(
                        blocks.AsSpan(blockIndex * SuperBlockBytes, SuperBlockBytes),
                        row.AsSpan(sb * SuperBlockElements, SuperBlockElements));
                }

                var sum = 0f;
                for (var i = 0; i < inputSize; i++)
                {
                    sum += row[i] * input[i];
                }
                output[o] = sum;
            }

            return output;
        }
    }
}
