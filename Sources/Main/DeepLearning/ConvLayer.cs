// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Kernels;
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

            var kData = new TensorStorage<float>(outChannels * _kernelSizePerOutput, clearMemory: false);
            InitializeKernels(kData.AsSpan(), _kernelSizePerOutput);

            Kernels = new AutogradNode(
                kData,
                new TensorShape(outChannels, _kernelSizePerOutput),
                requiresGrad: true);
        }

        public AutogradNode Kernels { get; }

        /// <summary>
        /// Optional per-channel bias. Null for layers without bias (e.g., Conv before BN).
        /// Populated by <see cref="LoadParameters(ReadOnlySpan{float}, ReadOnlySpan{float})"/>.
        /// </summary>
        public AutogradNode? Bias { get; private set; }

        public bool IsTraining { get; private set; } = true;

        public int InferenceInputSize => _inputSize;

        public int InferenceOutputSize => _outputSize;

        public void Train() => IsTraining = true;

        public void Eval()
        {
            IsTraining = false;
            PrepareInference();
        }

        public void PrepareInference()
        {
            // Direct inference convolution uses Kernels.DataView directly — no cache needed.
        }

        /// <summary>
        /// Loads pre-trained kernel weights. Layout: [outChannels, inChannels * kH * kW].
        /// ONNX Conv weight layout matches this exactly (NCHW ordering).
        /// </summary>
        public void LoadParameters(ReadOnlySpan<float> kernels)
        {
            var target = Kernels.DataView.AsSpan();

            if (kernels.Length != target.Length)
            {
                throw new ArgumentException(
                    $"Kernel size mismatch: expected {target.Length}, got {kernels.Length}.",
                    nameof(kernels));
            }

            kernels.CopyTo(target);
        }

        /// <summary>
        /// Loads kernel weights and per-channel bias.
        /// </summary>
        public void LoadParameters(ReadOnlySpan<float> kernels, ReadOnlySpan<float> bias)
        {
            LoadParameters(kernels);

            if (Bias == null)
            {
                var bData = new TensorStorage<float>(_outC, clearMemory: true);
                Bias = new AutogradNode(bData, new TensorShape(_outC), requiresGrad: true);
            }

            var bTarget = Bias.DataView.AsSpan();

            if (bias.Length != bTarget.Length)
            {
                throw new ArgumentException(
                    $"Bias size mismatch: expected {bTarget.Length}, got {bias.Length}.",
                    nameof(bias));
            }

            bias.CopyTo(bTarget);
        }

        public void InvalidateParameterCaches()
        {
            // No cached transformed weights.
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.Conv2D(graph, input, Kernels, _inC, _outC, _h, _w, _k);
            // Note: bias is applied in ForwardInference. Training path does not yet apply
            // conv bias via autograd — add TensorMath.AddBiasNchw when training with bias is needed.
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Kernels;
            if (Bias != null) yield return Bias;
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(Kernels.Shape.D0);
            bw.Write(Kernels.Shape.D1);

            foreach (var val in Kernels.DataView.AsReadOnlySpan())
            {
                bw.Write(val);
            }

            bw.Write(Bias != null ? 1 : 0);

            if (Bias != null)
            {
                foreach (var val in Bias.DataView.AsReadOnlySpan())
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

            var kSpan = Kernels.DataView.AsSpan();

            for (var i = 0; i < kSpan.Length; i++)
            {
                kSpan[i] = br.ReadSingle();
            }

            var hasBias = br.ReadInt32();

            if (hasBias == 1)
            {
                var bData = new TensorStorage<float>(_outC, clearMemory: false);
                Bias = new AutogradNode(bData, new TensorShape(_outC), requiresGrad: true);
                var bSpan = Bias.DataView.AsSpan();

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
            Conv2DKernels.ForwardValidNchw(
                input,
                Kernels.DataView.AsReadOnlySpan(),
                output,
                _inC, _outC, _h, _w, _k);

            // Apply per-channel bias if present
            if (Bias != null)
            {
                ApplyBiasNchw(output, Bias.DataView.AsReadOnlySpan(), _outC, _outH, _outW);
            }
        }

        public void ForwardInferencePrepared(ReadOnlySpan<float> input, Span<float> output)
        {
            ForwardInference(input, output);
        }

        public void Dispose()
        {
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
