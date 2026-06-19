// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Spatial batch normalisation for conv feature maps <c>[N, C, H, W]</c> — the standard CNN
    /// stabiliser. Normalises per channel over the N·H·W elements (one γ/β + running mean/var per
    /// channel), letting deeper conv stacks train faster and more reliably. Mirrors
    /// <see cref="BatchNorm1D"/> (which is per-feature over N); place it between conv and the activation.
    /// </summary>
    public sealed class BatchNorm2D : IModule
    {
        private readonly int _channels;
        private readonly TensorStorage<float> _inferenceScale;   // γ / √(var+eps), per channel
        private readonly TensorStorage<float> _inferenceShift;   // β − mean·scale, per channel
        private bool _inferenceCacheValid;

        private AutogradNode? _gammaNode;
        private AutogradNode? _betaNode;

        public BatchNorm2D(int channels)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);

            _channels = channels;

            Gamma = new Parameter(new TensorShape(channels), requiresGrad: true, clearData: false);
            Gamma.DataSpan.Fill(1f);
            Beta = new Parameter(new TensorShape(channels), requiresGrad: true, clearData: true);

            RunningMean = new TensorStorage<float>(channels, clearMemory: true);
            RunningVar = new TensorStorage<float>(channels, clearMemory: false);
            RunningVar.AsSpan().Fill(1f);

            _inferenceScale = new TensorStorage<float>(channels, clearMemory: false);
            _inferenceShift = new TensorStorage<float>(channels, clearMemory: false);
        }

        public Parameter Gamma
        {
            get;
        }
        public Parameter Beta
        {
            get;
        }
        public TensorStorage<float> RunningMean
        {
            get;
        }
        public TensorStorage<float> RunningVar
        {
            get;
        }
        public float Momentum { get; set; } = 0.1f;
        public float Eps { get; set; } = 1e-5f;
        public bool IsTraining { get; private set; } = true;

        public void Train()
        {
            IsTraining = true;
            _inferenceCacheValid = false;
        }

        public void Eval()
        {
            IsTraining = false;
            PrepareInference();
        }

        public void PrepareInference()
        {
            if (_inferenceCacheValid)
            {
                return;
            }

            var scale = _inferenceScale.AsSpan();
            var shift = _inferenceShift.AsSpan();

            TensorPrimitives.Add(RunningVar.AsReadOnlySpan(), Eps, scale);
            TensorPrimitives.ReciprocalSqrt(scale, scale);
            TensorPrimitives.Multiply(scale, Gamma.DataReadOnlySpan, scale);     // γ / √(var+eps)
            TensorPrimitives.Multiply(RunningMean.AsReadOnlySpan(), scale, shift);
            TensorPrimitives.Subtract(Beta.DataReadOnlySpan, shift, shift);       // β − mean·scale

            _inferenceCacheValid = true;
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            _gammaNode ??= Gamma.AsNode();
            _betaNode ??= Beta.AsNode();
            return ComputationGraph.BatchNorm2DOp(
                graph, input, _gammaNode, _betaNode, RunningMean, RunningVar, Momentum, Eps, IsTraining);
        }

        /// <summary>
        /// Inference shim for a <b>single</b> channel-major image <c>[C, H·W]</c> (length a multiple of
        /// the channel count); applies the cached per-channel scale/shift. For batched inference use
        /// <see cref="Forward"/> in <see cref="Eval"/> mode.
        /// </summary>
        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            if (!_inferenceCacheValid)
            {
                PrepareInference();
            }
            if (input.Length % _channels != 0)
            {
                throw new ArgumentException("Input length is not divisible by the channel count.", nameof(input));
            }
            if (output.Length < input.Length)
            {
                throw new ArgumentException("Output span is too small.", nameof(output));
            }

            var hw = input.Length / _channels;
            var scale = _inferenceScale.AsReadOnlySpan();
            var shift = _inferenceShift.AsReadOnlySpan();
            for (var c = 0; c < _channels; c++)
            {
                var outB = output.Slice(c * hw, hw);
                TensorPrimitives.Multiply(input.Slice(c * hw, hw), scale[c], outB);
                TensorPrimitives.Add(outB, shift[c], outB);
            }
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Gamma.AsNode();
            yield return Beta.AsNode();
        }

        public IEnumerable<Parameter> TrainableParameters()
        {
            yield return Gamma;
            yield return Beta;
        }

        public void InvalidateParameterCaches()
        {
            _inferenceCacheValid = false;
            if (!IsTraining)
            {
                PrepareInference();
            }
        }

        public void Save(BinaryWriter bw)
        {
            ArgumentNullException.ThrowIfNull(bw);
            bw.Write(_channels);
            foreach (var v in Gamma.DataReadOnlySpan)
            {
                bw.Write(v);
            }
            foreach (var v in Beta.DataReadOnlySpan)
            {
                bw.Write(v);
            }
            foreach (var v in RunningMean.AsReadOnlySpan())
            {
                bw.Write(v);
            }
            foreach (var v in RunningVar.AsReadOnlySpan())
            {
                bw.Write(v);
            }
        }

        public void Load(BinaryReader br)
        {
            ArgumentNullException.ThrowIfNull(br);
            var c = br.ReadInt32();
            if (c != _channels)
            {
                throw new Exception($"Channel count mismatch: {c} vs {_channels}");
            }

            var g = Gamma.DataSpan;
            for (var i = 0; i < _channels; i++)
            {
                g[i] = br.ReadSingle();
            }
            var b = Beta.DataSpan;
            for (var i = 0; i < _channels; i++)
            {
                b[i] = br.ReadSingle();
            }
            var rm = RunningMean.AsSpan();
            for (var i = 0; i < _channels; i++)
            {
                rm[i] = br.ReadSingle();
            }
            var rv = RunningVar.AsSpan();
            for (var i = 0; i < _channels; i++)
            {
                rv[i] = br.ReadSingle();
            }

            _inferenceCacheValid = false;
            if (!IsTraining)
            {
                PrepareInference();
            }
        }

        public void Dispose()
        {
            _gammaNode?.Dispose();
            _betaNode?.Dispose();
            Gamma.Dispose();
            Beta.Dispose();
            RunningMean.Dispose();
            RunningVar.Dispose();
            _inferenceScale.Dispose();
            _inferenceShift.Dispose();
        }
    }
}
