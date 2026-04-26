// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class LinearLayer : IModule, IInferenceShapeProvider
    {
        private readonly int _inputSize;
        private readonly int _outputSize;

        // Preallocated once. Rebuilt in-place. No TensorFactory.Materialize in inference hot path.
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

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.Linear(graph, input, Weights, Bias);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Weights;
            yield return Bias;
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

        private void RebuildTransposedWeightsInPlace()
        {
            var src = Weights.DataView.AsReadOnlySpan();
            var dst = _weightsTransposed.AsSpan();

            // Weights layout: [input, output]
            // Transposed cache layout: [output, input]
            for (var i = 0; i < _inputSize; i++)
            {
                var srcBase = i * _outputSize;

                for (var j = 0; j < _outputSize; j++)
                {
                    dst[j * _inputSize + i] = src[srcBase + j];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void LinearInferenceSimd(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsT,
            ReadOnlySpan<float> bias,
            Span<float> output)
        {
            var inputSize = input.Length;
            var outputSize = output.Length;

            for (var j = 0; j < outputSize; j++)
            {
                var wRow = weightsT.Slice(j * inputSize, inputSize);
                output[j] = TensorPrimitives.Dot(input, wRow) + bias[j];
            }
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            if (!_inferenceCacheValid)
            {
                PrepareInference();
            }

            if (input.Length % _inputSize != 0)
            {
                throw new ArgumentException("Input length is not divisible by layer input size.", nameof(input));
            }

            var batchSize = input.Length / _inputSize;
            var expectedOutputLength = batchSize * _outputSize;

            if (output.Length < expectedOutputLength)
            {
                throw new ArgumentException("Output span is too small for LinearLayer inference.", nameof(output));
            }

            var wTSpan = _weightsTransposed.AsReadOnlySpan();
            var bSpan = Bias.DataView.AsReadOnlySpan();

            if (batchSize == 1)
            {
                LinearInferenceSimd(
                    input,
                    wTSpan,
                    bSpan,
                    output.Slice(0, _outputSize));

                return;
            }

            for (var b = 0; b < batchSize; b++)
            {
                var inSlice = input.Slice(b * _inputSize, _inputSize);
                var outSlice = output.Slice(b * _outputSize, _outputSize);

                LinearInferenceSimd(
                    inSlice,
                    wTSpan,
                    bSpan,
                    outSlice);
            }
        }

        public void Dispose()
        {
            Weights.Dispose();
            Bias.Dispose();
            _weightsTransposed.Dispose();
        }
    }
}