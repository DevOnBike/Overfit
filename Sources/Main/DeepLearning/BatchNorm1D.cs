// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class BatchNorm1D : IModule, IInferenceShapeProvider
    {
        private readonly int _numFeatures;
        private readonly TensorStorage<float> _inferenceScale;
        private readonly TensorStorage<float> _inferenceShift;
        private bool _inferenceCacheValid;

        public BatchNorm1D(int numFeatures)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numFeatures);

            _numFeatures = numFeatures;

            Gamma = new AutogradNode(
                new TensorStorage<float>(numFeatures, clearMemory: false),
                new TensorShape(numFeatures),
                requiresGrad: true);

            Gamma.DataView.AsSpan().Fill(1f);

            Beta = new AutogradNode(
                new TensorStorage<float>(numFeatures, clearMemory: true),
                new TensorShape(numFeatures),
                requiresGrad: true);

            RunningMean = new TensorStorage<float>(numFeatures, clearMemory: true);
            RunningVar = new TensorStorage<float>(numFeatures, clearMemory: false);
            RunningVar.AsSpan().Fill(1f);

            _inferenceScale = new TensorStorage<float>(numFeatures, clearMemory: false);
            _inferenceShift = new TensorStorage<float>(numFeatures, clearMemory: false);
        }

        public AutogradNode Gamma { get; }

        public AutogradNode Beta { get; }

        public TensorStorage<float> RunningMean { get; }

        public TensorStorage<float> RunningVar { get; }

        public float Momentum { get; set; } = 0.1f;

        public float Eps { get; set; } = 1e-5f;

        public bool IsTraining { get; private set; } = true;

        public int InferenceInputSize => _numFeatures;

        public int InferenceOutputSize => _numFeatures;

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

            var scaleS = _inferenceScale.AsSpan();
            var shiftS = _inferenceShift.AsSpan();

            TensorPrimitives.Add(
                RunningVar.AsReadOnlySpan(),
                Eps,
                scaleS);

            TensorPrimitives.ReciprocalSqrt(scaleS, scaleS);

            TensorPrimitives.Multiply(
                scaleS,
                Gamma.DataView.AsReadOnlySpan(),
                scaleS);

            TensorPrimitives.Multiply(
                RunningMean.AsReadOnlySpan(),
                scaleS,
                shiftS);

            TensorPrimitives.Subtract(
                Beta.DataView.AsReadOnlySpan(),
                shiftS,
                shiftS);

            _inferenceCacheValid = true;
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return ComputationGraph.BatchNorm1DOp(
                graph,
                input,
                Gamma,
                Beta,
                RunningMean,
                RunningVar,
                Momentum,
                Eps,
                IsTraining);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Gamma;
            yield return Beta;
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(Gamma.DataView.Size);

            foreach (var val in Gamma.DataView.AsReadOnlySpan())
            {
                bw.Write(val);
            }

            foreach (var val in Beta.DataView.AsReadOnlySpan())
            {
                bw.Write(val);
            }

            foreach (var val in RunningMean.AsReadOnlySpan())
            {
                bw.Write(val);
            }

            foreach (var val in RunningVar.AsReadOnlySpan())
            {
                bw.Write(val);
            }
        }

        public void Load(BinaryReader br)
        {
            var len = br.ReadInt32();

            if (len != Gamma.DataView.Size)
            {
                throw new Exception($"Weight dimensions mismatch: {len} vs {Gamma.DataView.Size}");
            }

            var gSpan = Gamma.DataView.AsSpan();

            for (var i = 0; i < len; i++)
            {
                gSpan[i] = br.ReadSingle();
            }

            var bSpan = Beta.DataView.AsSpan();

            for (var i = 0; i < len; i++)
            {
                bSpan[i] = br.ReadSingle();
            }

            var rmSpan = RunningMean.AsSpan();

            for (var i = 0; i < len; i++)
            {
                rmSpan[i] = br.ReadSingle();
            }

            var rvSpan = RunningVar.AsSpan();

            for (var i = 0; i < len; i++)
            {
                rvSpan[i] = br.ReadSingle();
            }

            _inferenceCacheValid = false;

            if (!IsTraining)
            {
                PrepareInference();
            }
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            if (!_inferenceCacheValid)
            {
                PrepareInference();
            }

            var c = _numFeatures;

            if (input.Length % c != 0)
            {
                throw new ArgumentException("Input length is not divisible by BatchNorm feature count.", nameof(input));
            }

            if (output.Length < input.Length)
            {
                throw new ArgumentException("Output span is too small for BatchNorm inference.", nameof(output));
            }

            var batchSize = input.Length / c;
            var scaleS = _inferenceScale.AsReadOnlySpan();
            var shiftS = _inferenceShift.AsReadOnlySpan();

            for (var b = 0; b < batchSize; b++)
            {
                var inSlice = input.Slice(b * c, c);
                var outSlice = output.Slice(b * c, c);

                TensorPrimitives.MultiplyAdd(
                    inSlice,
                    scaleS,
                    shiftS,
                    outSlice);
            }
        }

        public void InvalidateParameterCaches()
        {
            _inferenceCacheValid = false;

            if (!IsTraining)
            {
                PrepareInference();
            }
        }

        public void Dispose()
        {
            Gamma.Dispose();
            Beta.Dispose();
            RunningMean.Dispose();
            RunningVar.Dispose();
            _inferenceScale.Dispose();
            _inferenceShift.Dispose();
        }
    }
}