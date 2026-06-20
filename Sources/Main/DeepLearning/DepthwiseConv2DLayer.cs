// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Depthwise convolution layer — one k×k filter per input channel (groups == channels), the cheap
    /// spatial half of a MobileNet-style depthwise-separable block (pair with a 1×1 <see cref="ConvLayer"/>
    /// to mix channels). Kernel shape <c>[channels, kSize·kSize]</c>; optional per-channel bias. SIMD
    /// inner kernel (AXPY over output rows); SAME with <c>padding = kSize/2</c>.
    /// </summary>
    public sealed class DepthwiseConv2DLayer : IModule
    {
        private readonly int _channels;
        private readonly int _h;
        private readonly int _w;
        private readonly int _k;
        private readonly int _padding;
        private readonly int _stride;
        private readonly int _outH;
        private readonly int _outW;

        private AutogradNode? _kernelsNode;
        private AutogradNode? _biasNode;
        private ComputationGraph? _inferenceGraph;

        public DepthwiseConv2DLayer(int channels, int h, int w, int kSize, int padding = 0, int stride = 1, bool useBias = false)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(h);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(w);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stride);
            ArgumentOutOfRangeException.ThrowIfNegative(padding);

            _channels = channels;
            _h = h;
            _w = w;
            _k = kSize;
            _padding = padding;
            _stride = stride;
            _outH = (h + 2 * padding - kSize) / stride + 1;
            _outW = (w + 2 * padding - kSize) / stride + 1;

            Kernels = new Parameter(new TensorShape(channels, kSize * kSize), requiresGrad: true, clearData: false);
            var stdDev = MathF.Sqrt(2f / (kSize * kSize));
            var span = Kernels.DataSpan;
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = MathUtils.NextGaussian() * stdDev;
            }

            if (useBias)
            {
                Bias = new Parameter(new TensorShape(channels), requiresGrad: true, clearData: true);
            }
        }

        public Parameter Kernels
        {
            get;
        }
        public Parameter? Bias
        {
            get;
        }
        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            _kernelsNode ??= Kernels.AsNode();
            if (Bias is not null)
            {
                _biasNode ??= Bias.AsNode();
            }
            return ComputationGraph.DepthwiseConv2DOp(
                graph, input, _kernelsNode, _channels, _h, _w, _k, _padding, _stride, _biasNode);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            _inferenceGraph ??= new ComputationGraph(checked(_channels * _outH * _outW * 4 + 1024));
            _inferenceGraph.Reset();
            _kernelsNode ??= Kernels.AsNode();
            if (Bias is not null)
            {
                _biasNode ??= Bias.AsNode();
            }

            var store = new TensorStorage<float>(input.Length, clearMemory: false);
            input.CopyTo(store.AsSpan());
            using var node = new AutogradNode(store, new TensorShape(1, _channels, _h, _w), requiresGrad: false);
            var outNode = ComputationGraph.DepthwiseConv2DOp(
                _inferenceGraph, node, _kernelsNode, _channels, _h, _w, _k, _padding, _stride, _biasNode);
            outNode.DataView.AsReadOnlySpan().CopyTo(output);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Kernels.AsNode();
            if (Bias is not null)
            {
                yield return Bias.AsNode();
            }
        }

        public IEnumerable<Parameter> TrainableParameters()
        {
            yield return Kernels;
            if (Bias is not null)
            {
                yield return Bias;
            }
        }

        public void InvalidateParameterCaches()
        {
        }

        public void Save(BinaryWriter bw)
        {
            ArgumentNullException.ThrowIfNull(bw);
            bw.Write(Kernels.Shape.D0);
            bw.Write(Kernels.Shape.D1);
            foreach (var v in Kernels.DataReadOnlySpan)
            {
                bw.Write(v);
            }
            bw.Write(Bias is not null ? 1 : 0);
            if (Bias is not null)
            {
                foreach (var v in Bias.DataReadOnlySpan)
                {
                    bw.Write(v);
                }
            }
        }

        public void Load(BinaryReader br)
        {
            ArgumentNullException.ThrowIfNull(br);
            var d0 = br.ReadInt32();
            var d1 = br.ReadInt32();
            if (d0 != Kernels.Shape.D0 || d1 != Kernels.Shape.D1)
            {
                throw new Exception("Depthwise kernel dimensions in file do not match the layer.");
            }
            var kSpan = Kernels.DataSpan;
            for (var i = 0; i < kSpan.Length; i++)
            {
                kSpan[i] = br.ReadSingle();
            }

            if (br.ReadInt32() == 1 && Bias is not null)
            {
                var bSpan = Bias.DataSpan;
                for (var i = 0; i < bSpan.Length; i++)
                {
                    bSpan[i] = br.ReadSingle();
                }
            }
        }

        public void Dispose()
        {
            _inferenceGraph?.Dispose();
            _kernelsNode?.Dispose();
            _biasNode?.Dispose();
            Kernels.Dispose();
            Bias?.Dispose();
        }
    }
}
