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
    public sealed class LinearLayer : IModule, IInferenceShapeProvider, IPreparedInferenceModule
    {
        private readonly int _inputSize;
        private readonly int _outputSize;

        // Layout: [output, input] — transposed for inference BLAS call
        private readonly TensorStorage<float> _weightsTransposed;
        private bool _inferenceCacheValid;

        // Cached view nodes — created once, reused across batches.
        // ownsDataStorage=false, ownsGradStorage=false: Parameter owns the storage.
        // Avoids per-batch AutogradNode heap allocation in Forward().
        private AutogradNode? _weightsNode;
        private AutogradNode? _biasNode;

        public LinearLayer(int inputSize, int outputSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputSize);

            _inputSize = inputSize;
            _outputSize = outputSize;

            Weights = new Parameter(
                new TensorShape(inputSize, outputSize),
                requiresGrad: true,
                clearData: false);

            var stdDev = MathF.Sqrt(2f / inputSize);
            var wSpan = Weights.DataSpan;

            for (var i = 0; i < wSpan.Length; i++)
            {
                wSpan[i] = MathUtils.NextGaussian() * stdDev;
            }

            Bias = new Parameter(new TensorShape(outputSize), requiresGrad: true, clearData: true);

            _weightsTransposed = new TensorStorage<float>(inputSize * outputSize, clearMemory: false);
        }

        /// <summary>Long-lived trainable weight matrix [inputSize, outputSize]. Owned by the layer.</summary>
        public Parameter Weights { get; }

        /// <summary>Long-lived trainable bias vector [outputSize]. Owned by the layer.</summary>
        public Parameter Bias { get; }

        public bool IsTraining { get; private set; } = true;

        public int InferenceInputSize => _inputSize;

        public int InferenceOutputSize => _outputSize;

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
            if (!_inferenceCacheValid)
            {
                RebuildTransposedWeightsInPlace();
                _inferenceCacheValid = true;
            }
        }

        /// <summary>
        /// Loads pre-trained weights from ONNX or another external source.
        /// Weights must be in [inputSize, outputSize] layout.
        /// PyTorch exports as [outputSize, inputSize] — transpose before calling.
        /// </summary>
        public void LoadParameters(ReadOnlySpan<float> weights, ReadOnlySpan<float> bias)
        {
            Weights.LoadData(weights);
            Bias.LoadData(bias);

            _inferenceCacheValid = false;

            if (!IsTraining)
            {
                PrepareInference();
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

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            // Cached view nodes: created once per layer lifetime, reused each batch.
            // Eliminates per-batch AutogradNode heap allocation (was: 2 new objects × 937 batches/epoch).
            // ownsDataStorage=false, ownsGradStorage=false — Parameter owns the storage.
            // Tape holds references to these nodes; they live until layer.Dispose().
            _weightsNode ??= Weights.AsNode();
            _biasNode    ??= Bias.AsNode();

            return ComputationGraph.LinearOp(graph, input, _weightsNode, _biasNode);
        }

        /// <summary>
        /// IModule compatibility shim — wraps Parameter as AutogradNode for existing optimizers.
        /// The returned nodes share Parameter's grad storage — ZeroGrad on the node zeroes Parameter.Grad.
        /// Prefer <see cref="TrainableParameters"/> for new code (Etap 6 optimizer migration).
        /// </summary>
        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Weights.AsNode();
            yield return Bias.AsNode();
        }

        /// <summary>
        /// Returns the <see cref="Parameter"/> objects owned by this layer.
        /// Used by optimizers after Etap 6 migration to <c>IEnumerable&lt;Parameter&gt;</c>.
        /// </summary>
        public IEnumerable<Parameter> TrainableParameters()
        {
            yield return Weights;
            yield return Bias;
        }

        public void Save(BinaryWriter bw)
        {
            Weights.Save(bw);
            Bias.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            Weights.Load(br);
            Bias.Load(br);

            _inferenceCacheValid = false;

            if (!IsTraining)
            {
                PrepareInference();
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
                throw new FileNotFoundException($"Weight file not found: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);
            Load(br);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            PrepareInference();
            LinearKernels.Forward(
                input,
                Weights.DataReadOnlySpan,
                _weightsTransposed.AsReadOnlySpan(),
                Bias.DataReadOnlySpan,
                output,
                _inputSize,
                _outputSize);
        }

        public void ForwardInferencePrepared(ReadOnlySpan<float> input, Span<float> output)
        {
            LinearKernels.Forward(
                input,
                Weights.DataReadOnlySpan,
                _weightsTransposed.AsReadOnlySpan(),
                Bias.DataReadOnlySpan,
                output,
                _inputSize,
                _outputSize);
        }

        public void Dispose()
        {
            _weightsNode?.Dispose();
            _biasNode?.Dispose();
            Weights.Dispose();
            Bias.Dispose();
            _weightsTransposed.Dispose();
        }

        private void RebuildTransposedWeightsInPlace()
        {
            LinearKernels.TransposeInputOutputToOutputInput(
                Weights.DataReadOnlySpan,
                _weightsTransposed.AsSpan(),
                _inputSize,
                _outputSize);
        }
    }
}
