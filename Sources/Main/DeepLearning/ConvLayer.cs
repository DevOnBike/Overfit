// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

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
            // No per-call cache is required.
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
                var inputBatchOffset = n * _inputSize;
                var outputBatchOffset = n * _outputSize;

                ForwardInferenceSingleBatch(
                    input.Slice(inputBatchOffset, _inputSize),
                    output.Slice(outputBatchOffset, _outputSize),
                    kernels);
            }
        }

        private void ForwardInferenceSingleBatch(
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