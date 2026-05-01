// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Parameters;
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
        private readonly int _padding;
        private readonly int _stride;

        // Cached kernel view node — created once, eliminates per-batch heap allocation.
        private AutogradNode? _kernelsNode;

        /// <summary>
        /// Creates a ConvLayer with padding=0, stride=1 (VALID convolution).
        /// </summary>
        public ConvLayer(
            int inChannels,
            int outChannels,
            int h,
            int w,
            int kSize)
            : this(inChannels, outChannels, h, w, kSize, padding: 0, stride: 1)
        {
        }

        /// <summary>
        /// Creates a ConvLayer with explicit padding and stride.
        /// Enables SAME-style convolution (padding = kSize/2) and strided convolution.
        /// </summary>
        public ConvLayer(
            int inChannels,
            int outChannels,
            int h,
            int w,
            int kSize,
            int padding,
            int stride)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inChannels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outChannels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(h);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(w);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stride);
            ArgumentOutOfRangeException.ThrowIfNegative(padding);

            _inC = inChannels;
            _outC = outChannels;
            _h = h;
            _w = w;
            _k = kSize;
            _padding = padding;
            _stride = stride;
            _outH = (h + 2 * padding - kSize) / stride + 1;
            _outW = (w + 2 * padding - kSize) / stride + 1;
            _inputSize = inChannels * h * w;
            _outputSize = outChannels * _outH * _outW;
            _kernelSizePerOutput = inChannels * kSize * kSize;

            Kernels = new Parameter(
                new TensorShape(outChannels, _kernelSizePerOutput),
                requiresGrad: true,
                clearData: false);

            InitializeKernels(Kernels.DataSpan, _kernelSizePerOutput);
        }

        public Parameter Kernels { get; }

        /// <summary>
        /// Optional per-channel bias. Null for layers without bias (e.g., Conv before BN).
        /// Populated by <see cref="LoadParameters(ReadOnlySpan{float}, ReadOnlySpan{float})"/>.
        /// </summary>
        public Parameter? Bias { get; private set; }

        public bool IsTraining { get; private set; } = true;

        public int InferenceInputSize
        {
            get
            {
                return _inputSize;
            }
        }

        public int InferenceOutputSize
        {
            get
            {
                return _outputSize;
            }
        }

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
            // Direct inference convolution uses Kernels.DataView directly â€” no cache needed.
        }

        /// <summary>
        /// Loads pre-trained kernel weights. Layout: [outChannels, inChannels * kH * kW].
        /// ONNX Conv weight layout matches this exactly (NCHW ordering).
        /// </summary>
        public void LoadParameters(ReadOnlySpan<float> kernels)
        {
            Kernels.LoadData(kernels);
        }

        /// <summary>
        /// Loads kernel weights and per-channel bias.
        /// </summary>
        public void LoadParameters(ReadOnlySpan<float> kernels, ReadOnlySpan<float> bias)
        {
            Kernels.LoadData(kernels);

            if (Bias == null)
            {
                Bias = new Parameter(new TensorShape(_outC), requiresGrad: true, clearData: true);
            }

            Bias.LoadData(bias);
        }

        public void InvalidateParameterCaches()
        {
            // No cached transformed weights.
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            _kernelsNode ??= Kernels.AsNode();
            return ComputationGraph.Conv2DOp(graph, input, _kernelsNode, _inC, _outC, _h, _w, _k);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Kernels.AsNode();
            if (Bias != null)
            {
                yield return Bias.AsNode();
            }
        }

        public IEnumerable<Parameter> TrainableParameters()
        {
            yield return Kernels;
            if (Bias != null)
            {
                yield return Bias;
            }
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(Kernels.Shape.D0);
            bw.Write(Kernels.Shape.D1);

            foreach (var val in Kernels.DataReadOnlySpan)
            {
                bw.Write(val);
            }

            bw.Write(Bias != null ? 1 : 0);

            if (Bias != null)
            {
                foreach (var val in Bias.DataReadOnlySpan)
                {
                    bw.Write(val);
                }
            }
        }

        public void Load(BinaryReader br)
        {
            var rows = br.ReadInt32();
            var cols = br.ReadInt32();

            if (rows != Kernels.Shape.D0 || cols != Kernels.Shape.D1)
            {
                throw new Exception("Kernel dimensions in file do not match the ConvLayer architecture.");
            }

            var kSpan = Kernels.DataSpan;

            for (var i = 0; i < kSpan.Length; i++)
            {
                kSpan[i] = br.ReadSingle();
            }

            var hasBias = br.ReadInt32();

            if (hasBias == 1)
            {
                if (Bias == null)
                {
                    Bias = new Parameter(new TensorShape(_outC), requiresGrad: true, clearData: false);
                }

                var bSpan = Bias.DataSpan;

                for (var i = 0; i < _outC; i++)
                {
                    bSpan[i] = br.ReadSingle();
                }
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

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            if (_padding == 0 && _stride == 1)
            {
                Conv2DKernels.ForwardValidNchw(
                    input, Kernels.DataReadOnlySpan, output,
                    _inC, _outC, _h, _w, _k);
            }
            else
            {
                Conv2DKernels.ForwardNchw(
                    input, Kernels.DataReadOnlySpan, output,
                    batchSize: 1, _inC, _outC, _h, _w, _k,
                    _padding, _stride);
            }

            if (Bias != null)
            {
                ApplyBiasNchw(output, Bias.DataReadOnlySpan, _outC, _outH, _outW);
            }
        }

        public void ForwardInferencePrepared(ReadOnlySpan<float> input, Span<float> output)
        {
            ForwardInference(input, output);
        }

        public void Dispose()
        {
            _kernelsNode?.Dispose();
            Kernels.Dispose();
            Bias?.Dispose();
        }

        private static void ApplyBiasNchw(Span<float> output, ReadOnlySpan<float> bias, int outC, int outH, int outW)
        {
            var spatialSize = outH * outW;
            for (var c = 0; c < outC; c++)
            {
                var b = bias[c];
                var channelSlice = output.Slice(c * spatialSize, spatialSize);
                for (var i = 0; i < channelSlice.Length; i++)
                {
                    channelSlice[i] += b;
                }
            }
        }

        private static void InitializeKernels(Span<float> span, int fanIn)
        {
            var stdDev = MathF.Sqrt(2f / fanIn);
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = MathUtils.NextGaussian() * stdDev;
            }
        }
    }
}
