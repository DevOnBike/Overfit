// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Maths;
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

        /// <summary>
        /// The kernel matrix repacked into MR-major micro-panels for the conv GEMM, built once when the
        /// layer enters inference mode. Null in training mode and on hardware that does not take the
        /// packed path. See <see cref="PrepareInference"/>.
        /// </summary>
        private TensorStorage<float>? _kernelsPacked;
        private readonly int _padding;
        private readonly int _stride;

        // Cached kernel / bias view nodes — created once, eliminate per-batch heap allocation.
        private AutogradNode? _kernelsNode;
        private AutogradNode? _biasNode;

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
        /// Set <paramref name="useBias"/> to train a per-output-channel bias (zero-initialised);
        /// the training/graph path now applies padding, stride, and bias with correct backward.
        /// </summary>
        public ConvLayer(
            int inChannels,
            int outChannels,
            int h,
            int w,
            int kSize,
            int padding,
            int stride,
            bool useBias = false)
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

            if (useBias)
            {
                Bias = new Parameter(new TensorShape(_outC), requiresGrad: true, clearData: true);
            }
        }

        public Parameter Kernels
        {
            get;
        }

        /// <summary>
        /// Multiply-accumulate inputs per output element: <c>inChannels * kernelSize * kernelSize</c>, which is
        /// the im2col contraction length K for this layer.
        ///
        /// <para>Exposed so a profiler can turn a per-node millisecond figure into an achieved GFLOP/s figure
        /// (<c>2 * InferenceOutputSize * this / seconds</c>). A ratio against another engine says who is faster;
        /// a fraction of the machine roofline says whether headroom exists, and only the second one decides
        /// whether work is worth starting. Internal because it is a shape detail, not a contract.</para>
        /// </summary>
        internal int KernelElementsPerOutput
        {
            get
            {
                return _kernelSizePerOutput;
            }
        }

        /// <summary>
        /// Optional per-channel bias. Null for layers without bias (e.g., Conv before BN).
        /// Populated by <see cref="LoadParameters(ReadOnlySpan{float}, ReadOnlySpan{float})"/>.
        /// </summary>
        public Parameter? Bias
        {
            get; private set;
        }

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

            // The packed kernels are an inference-mode artefact and go stale the moment a training
            // step touches the weights. Releasing them here also returns the second copy's memory.
            ReleasePackedKernels();
        }

        public void Eval()
        {
            IsTraining = false;
            PrepareInference();
        }

        /// <summary>
        /// Builds the MR-major repack of the kernel matrix, once, for the conv GEMM to sweep.
        ///
        /// <para><b>Why the pack belongs here and not in the kernel.</b> The GEMM's micro-kernel needs the
        /// MR values of one k-step side by side; read from the <c>[M, K]</c> matrix they are <c>K * 4</c>
        /// bytes apart, which is 18 KB on VGG-16's deepest layers. Packing fixes that, and <b>a
        /// convolution's A matrix is its own weights, so it never changes between inferences</b> — the
        /// pack is a load-time cost, not a per-call one. Measured with the pack done per call it was worth
        /// 4.1% on VGG-16 <i>net of paying for it every time</i>, and the per-layer split showed the whole
        /// of that cost landing on the layers with small N (conv13 lost 16.7% while conv6 gained 11.4%).
        /// Done here, those layers keep the gain and pay nothing.</para>
        ///
        /// <para><b>It costs a second copy of the kernel weights</b> — 58.8 MB across VGG-16's convolution
        /// stack — held only while the layer is in inference mode. <see cref="Train"/> releases it. That
        /// is a deliberate trade and it is the same shape as the one `XC-82` objects to for the dense
        /// layer; the difference is that this copy is on the hot path and measured to pay.</para>
        /// </summary>
        public void PrepareInference()
        {
            if (_kernelsPacked != null || !UsesPackedKernels())
            {
                return;
            }

            var packed = new TensorStorage<float>(
                Conv2DGemmKernels.PackedKernelLength(_outC, _kernelSizePerOutput),
                clearMemory: false);

            Conv2DGemmKernels.PackKernels(
                Kernels.DataReadOnlySpan, packed.AsSpan(), _outC, _kernelSizePerOutput);

            _kernelsPacked = packed;
        }

        /// <summary>
        /// Whether this layer's shape reaches the packed conv-GEMM path. Mirrors the branch in
        /// <c>Conv2DKernels</c>: the single-channel 3x3 case keeps its own vectorised kernel and never
        /// touches the GEMM, so packing for it would allocate a copy nothing reads.
        /// </summary>
        private bool UsesPackedKernels()
        {
            return Conv2DGemmKernels.UsePackedA
                && Conv2DGemmKernels.IsSupported
                && !(_inC == 1 && _k == 3);
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
            // The MR-major repack is derived from the kernel weights, so it is exactly the cache this
            // method exists to drop. Rebuilt on the next PrepareInference.
            ReleasePackedKernels();
        }

        private void ReleasePackedKernels()
        {
            _kernelsPacked?.Dispose();
            _kernelsPacked = null;
        }

        public AutogradNode Forward(ComputationGraph? graph, AutogradNode input)
        {
            _kernelsNode ??= Kernels.AsNode();
            if (Bias != null)
            {
                _biasNode ??= Bias.AsNode();
            }
            return ComputationGraph.Conv2DOp(
                graph, input, _kernelsNode, _inC, _outC, _h, _w, _k, _padding, _stride, _biasNode);
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

        /// <summary>The packed kernels if this layer has them, otherwise empty.</summary>
        private ReadOnlySpan<float> PackedKernelSpan()
        {
            return _kernelsPacked == null ? default : _kernelsPacked.AsReadOnlySpan();
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            if (_padding == 0 && _stride == 1)
            {
                Conv2DKernels.ForwardValidNchw(
                    input, Kernels.DataReadOnlySpan, output,
                    _inC, _outC, _h, _w, _k,
                    PackedKernelSpan());
            }

            if (_padding != 0 || _stride != 1)
            {
                Conv2DKernels.ForwardNchw(
                    input, Kernels.DataReadOnlySpan, output,
                    batchSize: 1, _inC, _outC, _h, _w, _k,
                    _padding, _stride,
                    PackedKernelSpan());
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
            _biasNode?.Dispose();
            Kernels.Dispose();
            Bias?.Dispose();
            _kernelsPacked?.Dispose();
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
