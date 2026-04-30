// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Parameters;
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

        // Cached view nodes — eliminates per-batch heap allocation.
        private AutogradNode? _gammaNode;
        private AutogradNode? _betaNode;

        public BatchNorm1D(int numFeatures)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numFeatures);

            _numFeatures = numFeatures;

            Gamma = new Parameter(new TensorShape(numFeatures), requiresGrad: true, clearData: false);
            Gamma.DataSpan.Fill(1f);

            Beta = new Parameter(new TensorShape(numFeatures), requiresGrad: true, clearData: true);

            RunningMean = new TensorStorage<float>(numFeatures, clearMemory: true);
            RunningVar = new TensorStorage<float>(numFeatures, clearMemory: false);
            RunningVar.AsSpan().Fill(1f);

            _inferenceScale = new TensorStorage<float>(numFeatures, clearMemory: false);
            _inferenceShift = new TensorStorage<float>(numFeatures, clearMemory: false);
        }

        public Parameter Gamma { get; }

        public Parameter Beta { get; }

        public TensorStorage<float> RunningMean { get; }

        public TensorStorage<float> RunningVar { get; }

        public float Momentum { get; set; } = 0.1f;

        public float Eps { get; set; } = 1e-5f;

        public bool IsTraining { get; private set; } = true;

        public int InferenceInputSize
        {
            get
            {
                return _numFeatures;
            }
        }

        public int InferenceOutputSize
        {
            get
            {
                return _numFeatures;
            }
        }

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
                Gamma.DataReadOnlySpan,
                scaleS);

            TensorPrimitives.Multiply(
                RunningMean.AsReadOnlySpan(),
                scaleS,
                shiftS);

            TensorPrimitives.Subtract(
                Beta.DataReadOnlySpan,
                shiftS,
                shiftS);

            _inferenceCacheValid = true;
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            _gammaNode ??= Gamma.AsNode();
            _betaNode  ??= Beta.AsNode();

            return ComputationGraph.BatchNorm1DOp(
                graph,
                input,
                _gammaNode,
                _betaNode,
                RunningMean,
                RunningVar,
                Momentum,
                Eps,
                IsTraining);
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

        public void Save(BinaryWriter bw)
        {
            bw.Write(Gamma.Shape.Size);

            foreach (var val in Gamma.DataReadOnlySpan)
            {
                bw.Write(val);
            }

            foreach (var val in Beta.DataReadOnlySpan)
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

            if (len != Gamma.Shape.Size)
            {
                throw new Exception($"Weight dimensions mismatch: {len} vs {Gamma.Shape.Size}");
            }

            var gSpan = Gamma.DataSpan;

            for (var i = 0; i < len; i++)
            {
                gSpan[i] = br.ReadSingle();
            }

            var bSpan = Beta.DataSpan;

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