// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Kernels
{
    public sealed class SingleTokenProjectionKernelTests
    {
        [Fact]
        public void Project_WithBias_ComputesMatrixVectorProduct()
        {
            // input [2]
            var input = new float[] { 2f, 3f };

            // weights [inputSize=2, outputSize=3]
            // row 0: [1, 2, 3]
            // row 1: [4, 5, 6]
            var weights = new float[]
            {
                1f, 2f, 3f,
                4f, 5f, 6f
            };

            var bias = new float[] { 10f, 20f, 30f };
            var output = new float[3];

            SingleTokenProjectionKernel.Project(
                input,
                weights,
                bias,
                output,
                inputSize: 2,
                outputSize: 3);

            AssertClose(24f, output[0]); // 10 + 2*1 + 3*4
            AssertClose(39f, output[1]); // 20 + 2*2 + 3*5
            AssertClose(54f, output[2]); // 30 + 2*3 + 3*6
        }

        [Fact]
        public void ProjectWithoutBias_ComputesMatrixVectorProduct()
        {
            var input = new float[] { 2f, 3f };
            var weights = new float[]
            {
                1f, 2f,
                3f, 4f
            };
            var output = new float[] { 99f, 99f };

            SingleTokenProjectionKernel.ProjectWithoutBias(
                input,
                weights,
                output,
                inputSize: 2,
                outputSize: 2);

            AssertClose(11f, output[0]); // 2*1 + 3*3
            AssertClose(16f, output[1]); // 2*2 + 3*4
        }

        [Fact]
        public void Accumulate_AddsOntoExistingOutput()
        {
            var input = new float[] { 2f };
            var weights = new float[] { 3f, 4f };
            var output = new float[] { 10f, 20f };

            SingleTokenProjectionKernel.Accumulate(
                input,
                weights,
                output,
                inputSize: 1,
                outputSize: 2);

            AssertClose(16f, output[0]);
            AssertClose(28f, output[1]);
        }

        [Fact]
        public void Project_DoesNotUseStaleOutputValues()
        {
            var input = new float[] { 1f };
            var weights = new float[] { 7f };
            var bias = Array.Empty<float>();
            var output = new float[] { 123f };

            SingleTokenProjectionKernel.Project(
                input,
                weights,
                bias,
                output,
                inputSize: 1,
                outputSize: 1);

            AssertClose(7f, output[0]);
        }

        [Fact]
        public void ProjectSlice_ComputesOnlyRequestedOutputSlice()
        {
            var input = new float[] { 2f, 3f };

            // weights [2, 5]
            var weights = new float[]
            {
                1f, 2f, 3f, 4f, 5f,
                6f, 7f, 8f, 9f, 10f
            };

            var bias = new float[] { 100f, 200f, 300f, 400f, 500f };
            var output = new float[2];

            SingleTokenProjectionKernel.ProjectSlice(
                input,
                weights,
                bias,
                output,
                inputSize: 2,
                fullOutputSize: 5,
                outputOffset: 2,
                outputCount: 2);

            AssertClose(330f, output[0]); // 300 + 2*3 + 3*8
            AssertClose(435f, output[1]); // 400 + 2*4 + 3*9
        }

        [Fact]
        public void CopyToCachePosition_CopiesKeyAndValue()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 2,
                headDimension: 3);

            var source = new float[] { 1f, 2f, 3f };

            SingleTokenProjectionKernel.CopyToCachePosition(
                source,
                cache,
                layerIndex: 0,
                headIndex: 0,
                position: 0,
                copyToKey: true,
                copyToValue: true);

            cache.Advance();

            Assert.Equal(new float[] { 1f, 2f, 3f }, cache.GetKeyReadSpan(0, 0, 0, 1).ToArray());
            Assert.Equal(new float[] { 1f, 2f, 3f }, cache.GetValueReadSpan(0, 0, 0, 1).ToArray());
        }

        [Fact]
        public void CopyToCachePosition_CopiesOnlyRequestedSide()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 2,
                headDimension: 2);

            var source = new float[] { 7f, 8f };

            SingleTokenProjectionKernel.CopyToCachePosition(
                source,
                cache,
                layerIndex: 0,
                headIndex: 0,
                position: 0,
                copyToKey: true,
                copyToValue: false);

            cache.Advance();

            Assert.Equal(new float[] { 7f, 8f }, cache.GetKeyReadSpan(0, 0, 0, 1).ToArray());
            Assert.Equal(new float[] { 0f, 0f }, cache.GetValueReadSpan(0, 0, 0, 1).ToArray());
        }

        [Fact]
        public void Project_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                SingleTokenProjectionKernel.Project(
                    input: new float[1],
                    weightsInputOutput: new float[1],
                    bias: [],
                    output: new float[1],
                    inputSize: 0,
                    outputSize: 1));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                SingleTokenProjectionKernel.Project(
                    input: new float[1],
                    weightsInputOutput: new float[1],
                    bias: [],
                    output: new float[1],
                    inputSize: 1,
                    outputSize: 0));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenProjectionKernel.Project(
                    input: new float[0],
                    weightsInputOutput: new float[1],
                    bias: [],
                    output: new float[1],
                    inputSize: 1,
                    outputSize: 1));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenProjectionKernel.Project(
                    input: new float[1],
                    weightsInputOutput: new float[0],
                    bias: [],
                    output: new float[1],
                    inputSize: 1,
                    outputSize: 1));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenProjectionKernel.Project(
                    input: new float[1],
                    weightsInputOutput: new float[1],
                    bias: new float[0],
                    output: new float[0],
                    inputSize: 1,
                    outputSize: 1));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenProjectionKernel.Project(
                    input: new float[1],
                    weightsInputOutput: new float[1],
                    bias: new float[0],
                    output: new float[1],
                    inputSize: 1,
                    outputSize: 2));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenProjectionKernel.Project(
                    input: new float[1],
                    weightsInputOutput: new float[1],
                    bias: new float[1],
                    output: new float[1],
                    inputSize: 1,
                    outputSize: 2));
        }

        [Fact]
        public void ProjectSlice_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                SingleTokenProjectionKernel.ProjectSlice(
                    input: new float[1],
                    weightsInputOutput: new float[1],
                    bias: [],
                    output: new float[1],
                    inputSize: 1,
                    fullOutputSize: 2,
                    outputOffset: -1,
                    outputCount: 1));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                SingleTokenProjectionKernel.ProjectSlice(
                    input: new float[1],
                    weightsInputOutput: new float[2],
                    bias: [],
                    output: new float[1],
                    inputSize: 1,
                    fullOutputSize: 2,
                    outputOffset: 1,
                    outputCount: 2));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenProjectionKernel.ProjectSlice(
                    input: new float[1],
                    weightsInputOutput: new float[2],
                    bias: new float[1],
                    output: new float[1],
                    inputSize: 1,
                    fullOutputSize: 2,
                    outputOffset: 1,
                    outputCount: 1));
        }

        [Fact]
        public void CopyToCachePosition_InvalidArguments_Throw()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 1,
                headDimension: 2);

            Assert.Throws<ArgumentNullException>(() =>
                SingleTokenProjectionKernel.CopyToCachePosition(
                    source: new float[2],
                    cache: null!,
                    layerIndex: 0,
                    headIndex: 0,
                    position: 0,
                    copyToKey: true,
                    copyToValue: true));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenProjectionKernel.CopyToCachePosition(
                    source: new float[1],
                    cache,
                    layerIndex: 0,
                    headIndex: 0,
                    position: 0,
                    copyToKey: true,
                    copyToValue: true));
        }

        private static void AssertClose(float expected, float actual)
        {
            Assert.True(
                MathF.Abs(expected - actual) <= 1e-5f,
                $"Expected {expected}, actual {actual}.");
        }
    }
}
