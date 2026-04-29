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
    public sealed class LinearLayer : IModule, IInferenceShapeProvider, IPreparedInferenceModule
    {
        private readonly int _inputSize;
        private readonly int _outputSize;

        // Layout: [output, input] â€” transposed for inference BLAS call
        private readonly TensorStorage<float> _weightsTransposed;
        private bool _inferenceCacheValid;

        public LinearLayer(int inputSize, int outputSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputSize);

            _inputSize = inputSize;
            _outputSize = outputSize;

            var wData = new TensorStorage<float>(inputSize * outputSize, clearMemory: false);
            var stdDev = MathF.Sqrt(2f / inputSize);
            var wSpan = wData.AsSpan();

            for (var i = 0; i < wSpan.Length; i++)
            {
                wSpan[i] = MathUtils.NextGaussian() * stdDev;
            }

            Weights = new AutogradNode(
                wData,
                new TensorShape(inputSize, outputSize),
                requiresGrad: true);

            var bData = new TensorStorage<float>(outputSize, clearMemory: true);

            Bias = new AutogradNode(
                bData,
                new TensorShape(outputSize),
                requiresGrad: true);

            _weightsTransposed = new TensorStorage<float>(
                inputSize * outputSize,
                clearMemory: false);
        }

        public AutogradNode Weights { get; }

        public AutogradNode Bias { get; }

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
        /// Weights must be in [inputSize, outputSize] layout (row = input neuron, column = output neuron).
        /// PyTorch exports Linear weights as [outputSize, inputSize] â€” transpose before calling.
        /// </summary>
        public void LoadParameters(ReadOnlySpan<float> weights, ReadOnlySpan<float> bias)
        {
            var wTarget = Weights.DataView.AsSpan();
            if (weights.Length != wTarget.Length)
            {
                throw new ArgumentException(
                    $"Weight size mismatch: expected {wTarget.Length}, got {weights.Length}.",
                    nameof(weights));
            }

            weights.CopyTo(wTarget);

            var bTarget = Bias.DataView.AsSpan();
            if (bias.Length != bTarget.Length)
            {
                throw new ArgumentException(
                    $"Bias size mismatch: expected {bTarget.Length}, got {bias.Length}.",
                    nameof(bias));
            }

            bias.CopyTo(bTarget);

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
            return ComputationGraph.LinearOp(graph, input, Weights, Bias);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Weights;
            yield return Bias;
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(Weights.DataView.Size);

            foreach (var val in Weights.DataView.AsReadOnlySpan())
            {
                bw.Write(val);
            }

            bw.Write(Bias.DataView.Size);

            foreach (var val in Bias.DataView.AsReadOnlySpan())
            {
                bw.Write(val);
            }
        }

        public void Load(BinaryReader br)
        {
            var lenW = br.ReadInt32();

            if (lenW != Weights.DataView.Size)
            {
                throw new Exception("Weights mismatch");
            }

            var wSpan = Weights.DataView.AsSpan();

            for (var i = 0; i < lenW; i++)
            {
                wSpan[i] = br.ReadSingle();
            }

            var lenB = br.ReadInt32();

            if (lenB != Bias.DataView.Size)
            {
                throw new Exception("Bias mismatch");
            }

            var bSpan = Bias.DataView.AsSpan();

            for (var i = 0; i < lenB; i++)
            {
                bSpan[i] = br.ReadSingle();
            }

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
                Weights.DataView.AsReadOnlySpan(),
                _weightsTransposed.AsReadOnlySpan(),
                Bias.DataView.AsReadOnlySpan(),
                output,
                _inputSize,
                _outputSize);
        }

        public void ForwardInferencePrepared(ReadOnlySpan<float> input, Span<float> output)
        {
            // Cache guaranteed valid â€” called only after PrepareInference() has run.
            LinearKernels.Forward(
                input,
                Weights.DataView.AsReadOnlySpan(),
                _weightsTransposed.AsReadOnlySpan(),
                Bias.DataView.AsReadOnlySpan(),
                output,
                _inputSize,
                _outputSize);
        }

        public void Dispose()
        {
            Weights.Dispose();
            Bias.Dispose();
            _weightsTransposed.Dispose();
        }

        private void RebuildTransposedWeightsInPlace()
        {
            LinearKernels.TransposeInputOutputToOutputInput(
                Weights.DataView.AsReadOnlySpan(),
                _weightsTransposed.AsSpan(),
                _inputSize,
                _outputSize);
        }
    }
}
