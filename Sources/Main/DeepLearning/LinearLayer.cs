// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    ///     Implements a fully-connected Linear (Dense) Layer.
    ///     Features an optimized zero-allocation SIMD inference path through weight pre-transposition.
    /// </summary>
    public sealed class LinearLayer : IModule
    {

        // Inference buffer used to achieve Zero-Allocation during forward passes.
        private readonly AutogradNode _inferenceOutputNode;

        private readonly int _inputSize;
        private readonly int _outputSize;

        // Cached transposed weights for SIMD inference — shape: (outputSize, inputSize).
        // Sequential memory access in the inner loop ensures maximum SIMD utilization.
        private FastTensor<float> _weightsTransposed;

        public LinearLayer(int inputSize, int outputSize)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;

            var wData = new FastTensor<float>(inputSize, outputSize);
            var stdDev = MathF.Sqrt(2f / inputSize);
            var wSpan = wData.AsSpan();
            for (var i = 0; i < wSpan.Length; i++)
            {
                wSpan[i] = MathUtils.NextGaussian() * stdDev;
            }

            Weights = new AutogradNode(wData);
            Biases = new AutogradNode(new FastTensor<float>(outputSize));

            var outData = new FastTensor<float>(1, outputSize);
            _inferenceOutputNode = new AutogradNode(outData, false);
        }
        public AutogradNode Weights { get; }
        public AutogradNode Biases { get; }
        public bool IsTraining { get; private set; } = true;

        public void Train()
        {
            IsTraining = true;

            _weightsTransposed?.Dispose();
            _weightsTransposed = null;
        }

        public void Eval()
        {
            IsTraining = false;

            RebuildTransposedWeights();
        }

        /// <summary>
        ///     Performs the forward pass. Dispatches to a highly optimized SIMD kernel for single-row inference.
        /// </summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            if (graph == null || !IsTraining)
            {
                var batchSize = input.Data.GetDim(0);

                if (batchSize != 1)
                {
                    return TensorMath.Linear(null, input, Weights, Biases);
                }

                if (_weightsTransposed == null)
                {
                    RebuildTransposedWeights();
                }

                LinearInferenceSimd(
                input.Data.AsReadOnlySpan(),
                _weightsTransposed.AsReadOnlySpan(),
                Biases.Data.AsReadOnlySpan(),
                _inferenceOutputNode.Data.AsSpan());

                return _inferenceOutputNode;
            }

            if (_weightsTransposed != null)
            {
                _weightsTransposed.Dispose();
                _weightsTransposed = null;
            }

            return TensorMath.Linear(graph, input, Weights, Biases);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Weights;
            yield return Biases;
        }

        public void Save(BinaryWriter bw)
        {
            var wSpan = Weights.Data.AsReadOnlySpan();
            for (var i = 0; i < wSpan.Length; i++)
            {
                bw.Write(wSpan[i]);
            }

            var bSpan = Biases.Data.AsReadOnlySpan();
            for (var i = 0; i < bSpan.Length; i++)
            {
                bw.Write(bSpan[i]);
            }
        }

        public void Load(BinaryReader br)
        {
            var wSpan = Weights.Data.AsSpan();
            for (var i = 0; i < wSpan.Length; i++)
            {
                wSpan[i] = br.ReadSingle();
            }

            var bSpan = Biases.Data.AsSpan();
            for (var i = 0; i < bSpan.Length; i++)
            {
                bSpan[i] = br.ReadSingle();
            }

            if (!IsTraining)
            {
                RebuildTransposedWeights();
            }
        }

        public void Dispose()
        {
            Weights?.Dispose();
            Biases?.Dispose();
            _inferenceOutputNode?.Dispose();
            _weightsTransposed?.Dispose();
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

        /// <summary>
        ///     Transposes weights: W[inputSize, outputSize] → W_T[outputSize, inputSize].
        ///     This ensures that data for each output neuron lies sequentially in memory,
        /// </summary>
        private void RebuildTransposedWeights()
        {
            _weightsTransposed?.Dispose();
            _weightsTransposed = new FastTensor<float>(_outputSize, _inputSize);

            var src = Weights.Data.AsReadOnlySpan();
            var dst = _weightsTransposed.AsSpan();

            for (var i = 0; i < _inputSize; i++)
            {
                for (var j = 0; j < _outputSize; j++)
                {
                    dst[j * _inputSize + i] = src[i * _outputSize + j];
                }
            }
        }

        /// <summary>
        ///     Highly optimized SIMD inference kernel for pre-transposed weights.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void LinearInferenceSimd(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsT,
            ReadOnlySpan<float> bias,
            Span<float> output)
        {
            var inputSize = input.Length;
            var outputSize = output.Length;
            var vCount = Vector<float>.Count;

            for (var j = 0; j < outputSize; j++)
            {
                var sum = Vector<float>.Zero;
                var wRow = weightsT.Slice(j * inputSize, inputSize);
                var i = 0;

                for (; i <= inputSize - vCount; i += vCount)
                {
                    var vIn = new Vector<float>(input.Slice(i));
                    var vW = new Vector<float>(wRow.Slice(i));
                    sum += vIn * vW;
                }

                var scalarSum = Vector.Dot(sum, Vector<float>.One) + bias[j];

                for (; i < inputSize; i++)
                {
                    scalarSum += input[i] * wRow[i];
                }

                output[j] = scalarSum;
            }
        }
    }
}