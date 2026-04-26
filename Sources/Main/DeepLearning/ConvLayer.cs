// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class ConvLayer : IModule, IInferenceShapeProvider
    {
        private readonly int _inC;
        private readonly int _outC;
        private readonly int _h;
        private readonly int _w;
        private readonly int _k;
        private readonly int _outH;
        private readonly int _outW;
        private readonly int _inputSize;
        private readonly int _outputSize;
        private readonly int _kernelSizePerOutput;

        public ConvLayer(
            int inChannels,
            int outChannels,
            int h,
            int w,
            int kSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inChannels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outChannels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(h);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(w);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kSize);

            if (kSize > h || kSize > w)
            {
                throw new ArgumentException("Kernel size cannot be larger than input spatial dimensions.");
            }

            _inC = inChannels;
            _outC = outChannels;
            _h = h;
            _w = w;
            _k = kSize;
            _outH = h - kSize + 1;
            _outW = w - kSize + 1;
            _inputSize = inChannels * h * w;
            _outputSize = outChannels * _outH * _outW;
            _kernelSizePerOutput = inChannels * kSize * kSize;

            var kData = new TensorStorage<float>(
                outChannels * _kernelSizePerOutput,
                clearMemory: false);

            InitializeKernels(
                kData.AsSpan(),
                _kernelSizePerOutput);

            Kernels = new AutogradNode(
                kData,
                new TensorShape(outChannels, _kernelSizePerOutput),
                requiresGrad: true);
        }

        public AutogradNode Kernels { get; }

        public bool IsTraining { get; private set; } = true;

        public int InferenceInputSize => _inputSize;

        public int InferenceOutputSize => _outputSize;

        public void Train()
        {
            IsTraining = true;
        }

        public void Eval()
        {
            IsTraining = false;
            PrepareInference();
        }

        public void PrepareInference()
        {
            // Direct inference convolution uses Kernels.DataView directly.
            // No per-call cache required.
        }

        public AutogradNode Forward(
            ComputationGraph graph,
            AutogradNode input)
        {
            return TensorMath.Conv2D(
                graph,
                input,
                Kernels,
                _inC,
                _outC,
                _h,
                _w,
                _k);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Kernels;
        }

        public void InvalidateParameterCaches()
        {
            // No cached transformed weights currently.
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(Kernels.Shape.D0);
            bw.Write(Kernels.Shape.D1);

            foreach (var val in Kernels.DataView.AsReadOnlySpan())
            {
                bw.Write(val);
            }
        }

        public void Load(BinaryReader br)
        {
            var rows = br.ReadInt32();
            var cols = br.ReadInt32();

            if (rows != Kernels.Shape.D0 ||
                cols != Kernels.Shape.D1)
            {
                throw new Exception("Kernel dimensions in file do not match the ConvLayer architecture.");
            }

            var span = Kernels.DataView.AsSpan();

            for (var i = 0; i < span.Length; i++)
            {
                span[i] = br.ReadSingle();
            }
        }

        public void ForwardInference(
            ReadOnlySpan<float> input,
            Span<float> output)
        {
            if (input.Length % _inputSize != 0)
            {
                throw new ArgumentException(
                    "Input length is not divisible by ConvLayer inference input size.",
                    nameof(input));
            }

            var batchSize = input.Length / _inputSize;
            var expectedOutputLength = batchSize * _outputSize;

            if (output.Length < expectedOutputLength)
            {
                throw new ArgumentException(
                    "Output span is too small for ConvLayer inference.",
                    nameof(output));
            }

            var kernels = Kernels.DataView.AsReadOnlySpan();

            for (var n = 0; n < batchSize; n++)
            {
                var inputBatch = input.Slice(n * _inputSize, _inputSize);
                var outputBatch = output.Slice(n * _outputSize, _outputSize);

                if (_inC == 1 && _k == 3)
                {
                    ForwardInferenceSingleChannel3x3(
                        inputBatch,
                        outputBatch,
                        kernels);
                }
                else
                {
                    ForwardInferenceGenericSingleBatch(
                        inputBatch,
                        outputBatch,
                        kernels);
                }
            }
        }

        private void ForwardInferenceSingleChannel3x3(
            ReadOnlySpan<float> input,
            Span<float> output,
            ReadOnlySpan<float> kernels)
        {
            if (Vector.IsHardwareAccelerated &&
                _outW >= Vector<float>.Count)
            {
                ForwardInferenceSingleChannel3x3Vectorized(
                    input,
                    output,
                    kernels);

                return;
            }

            ForwardInferenceSingleChannel3x3Scalar(
                input,
                output,
                kernels);
        }

        private void ForwardInferenceSingleChannel3x3Vectorized(
            ReadOnlySpan<float> input,
            Span<float> output,
            ReadOnlySpan<float> kernels)
        {
            var vectorWidth = Vector<float>.Count;

            for (var oc = 0; oc < _outC; oc++)
            {
                var kernelBase = oc * 9;
                var outputChannelBase = oc * _outH * _outW;

                var k00 = new Vector<float>(kernels[kernelBase + 0]);
                var k01 = new Vector<float>(kernels[kernelBase + 1]);
                var k02 = new Vector<float>(kernels[kernelBase + 2]);
                var k10 = new Vector<float>(kernels[kernelBase + 3]);
                var k11 = new Vector<float>(kernels[kernelBase + 4]);
                var k12 = new Vector<float>(kernels[kernelBase + 5]);
                var k20 = new Vector<float>(kernels[kernelBase + 6]);
                var k21 = new Vector<float>(kernels[kernelBase + 7]);
                var k22 = new Vector<float>(kernels[kernelBase + 8]);

                for (var oy = 0; oy < _outH; oy++)
                {
                    var inputRow0 = oy * _w;
                    var inputRow1 = (oy + 1) * _w;
                    var inputRow2 = (oy + 2) * _w;
                    var outputRow = outputChannelBase + oy * _outW;

                    var ox = 0;

                    for (; ox <= _outW - vectorWidth; ox += vectorWidth)
                    {
                        var acc =
                            new Vector<float>(input.Slice(inputRow0 + ox, vectorWidth)) * k00 +
                            new Vector<float>(input.Slice(inputRow0 + ox + 1, vectorWidth)) * k01 +
                            new Vector<float>(input.Slice(inputRow0 + ox + 2, vectorWidth)) * k02 +
                            new Vector<float>(input.Slice(inputRow1 + ox, vectorWidth)) * k10 +
                            new Vector<float>(input.Slice(inputRow1 + ox + 1, vectorWidth)) * k11 +
                            new Vector<float>(input.Slice(inputRow1 + ox + 2, vectorWidth)) * k12 +
                            new Vector<float>(input.Slice(inputRow2 + ox, vectorWidth)) * k20 +
                            new Vector<float>(input.Slice(inputRow2 + ox + 1, vectorWidth)) * k21 +
                            new Vector<float>(input.Slice(inputRow2 + ox + 2, vectorWidth)) * k22;

                        acc.CopyTo(output.Slice(outputRow + ox, vectorWidth));
                    }

                    for (; ox < _outW; ox++)
                    {
                        output[outputRow + ox] =
                            input[inputRow0 + ox] * kernels[kernelBase + 0] +
                            input[inputRow0 + ox + 1] * kernels[kernelBase + 1] +
                            input[inputRow0 + ox + 2] * kernels[kernelBase + 2] +
                            input[inputRow1 + ox] * kernels[kernelBase + 3] +
                            input[inputRow1 + ox + 1] * kernels[kernelBase + 4] +
                            input[inputRow1 + ox + 2] * kernels[kernelBase + 5] +
                            input[inputRow2 + ox] * kernels[kernelBase + 6] +
                            input[inputRow2 + ox + 1] * kernels[kernelBase + 7] +
                            input[inputRow2 + ox + 2] * kernels[kernelBase + 8];
                    }
                }
            }
        }

        private void ForwardInferenceSingleChannel3x3Scalar(
            ReadOnlySpan<float> input,
            Span<float> output,
            ReadOnlySpan<float> kernels)
        {
            for (var oc = 0; oc < _outC; oc++)
            {
                var kernelBase = oc * 9;
                var outputChannelBase = oc * _outH * _outW;

                var k00 = kernels[kernelBase + 0];
                var k01 = kernels[kernelBase + 1];
                var k02 = kernels[kernelBase + 2];
                var k10 = kernels[kernelBase + 3];
                var k11 = kernels[kernelBase + 4];
                var k12 = kernels[kernelBase + 5];
                var k20 = kernels[kernelBase + 6];
                var k21 = kernels[kernelBase + 7];
                var k22 = kernels[kernelBase + 8];

                for (var oy = 0; oy < _outH; oy++)
                {
                    var inputRow0 = oy * _w;
                    var inputRow1 = (oy + 1) * _w;
                    var inputRow2 = (oy + 2) * _w;
                    var outputRow = outputChannelBase + oy * _outW;

                    for (var ox = 0; ox < _outW; ox++)
                    {
                        output[outputRow + ox] =
                            input[inputRow0 + ox] * k00 +
                            input[inputRow0 + ox + 1] * k01 +
                            input[inputRow0 + ox + 2] * k02 +
                            input[inputRow1 + ox] * k10 +
                            input[inputRow1 + ox + 1] * k11 +
                            input[inputRow1 + ox + 2] * k12 +
                            input[inputRow2 + ox] * k20 +
                            input[inputRow2 + ox + 1] * k21 +
                            input[inputRow2 + ox + 2] * k22;
                    }
                }
            }
        }

        private void ForwardInferenceGenericSingleBatch(
            ReadOnlySpan<float> input,
            Span<float> output,
            ReadOnlySpan<float> kernels)
        {
            for (var oc = 0; oc < _outC; oc++)
            {
                var kernelBase = oc * _kernelSizePerOutput;
                var outputChannelBase = oc * _outH * _outW;

                for (var oy = 0; oy < _outH; oy++)
                {
                    for (var ox = 0; ox < _outW; ox++)
                    {
                        var sum = 0f;

                        for (var ic = 0; ic < _inC; ic++)
                        {
                            var inputChannelBase = ic * _h * _w;
                            var kernelChannelBase = kernelBase + ic * _k * _k;

                            for (var ky = 0; ky < _k; ky++)
                            {
                                var inputRowBase = inputChannelBase + (oy + ky) * _w + ox;
                                var kernelRowBase = kernelChannelBase + ky * _k;

                                for (var kx = 0; kx < _k; kx++)
                                {
                                    sum += input[inputRowBase + kx] *
                                           kernels[kernelRowBase + kx];
                                }
                            }
                        }

                        output[outputChannelBase + oy * _outW + ox] = sum;
                    }
                }
            }
        }

        public void Dispose()
        {
            Kernels.Dispose();
        }

        private static void InitializeKernels(
            Span<float> span,
            int fanIn)
        {
            var stdDev = MathF.Sqrt(2f / fanIn);

            for (var i = 0; i < span.Length; i++)
            {
                span[i] = MathUtils.NextGaussian() * stdDev;
            }
        }

        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);

            Save(bw);
        }

        public void Load(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Model weights file not found: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);

            Load(br);
        }
    }
}