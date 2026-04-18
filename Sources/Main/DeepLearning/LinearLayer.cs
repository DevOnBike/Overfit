// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.DeepLearning.Diagnostics;
using DevOnBike.Overfit.Diagnostics.Contracts;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class LinearLayer : IModule
    {
        private AutogradNode _inferenceOutputNode;
        private int _inferenceOutputBatchSize = 1;
        private readonly int _inputSize;
        private readonly int _outputSize;
        private FastTensor<float> _weightsTransposed;

        public LinearLayer(int inputSize, int outputSize)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;

            var wData = new FastTensor<float>(inputSize, outputSize, clearMemory: false);
            var stdDev = MathF.Sqrt(2f / inputSize);
            var wSpan = wData.GetView().AsSpan();
            for (var i = 0; i < wSpan.Length; i++)
            {
                wSpan[i] = MathUtils.NextGaussian() * stdDev;
            }

            Weights = new AutogradNode(wData, requiresGrad: true);
            Biases = new AutogradNode(new FastTensor<float>(outputSize), requiresGrad: true);

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

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            if (IsTraining)
            {
                throw new InvalidOperationException("Layer must be in Eval mode.");
            }

            if (_weightsTransposed == null)
            {
                RebuildTransposedWeights();
            }

            using (new DiagnosticScope(
                   category: "DeepLearning",
                   name: "LinearLayer.ForwardInference",
                   phase: "forward_inference",
                   isTraining: false,
                   batchSize: 1,
                   featureCount: _outputSize,
                   inputElements: input.Length,
                   outputElements: output.Length))
            {
                LinearInferenceSimd(
                input,
                _weightsTransposed.GetView().AsReadOnlySpan(),
                Biases.DataView.AsReadOnlySpan(),
                output);
            }
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var batchSize = input.DataView.GetDim(0);

            var ctx = ModuleDiagnostics.Begin(
            moduleType: nameof(LinearLayer),
            phase: graph == null || !IsTraining ? "forward_eval" : "forward_train",
            isTraining: IsTraining,
            batchSize: batchSize,
            inputRows: batchSize,
            inputCols: _inputSize,
            outputRows: batchSize,
            outputCols: _outputSize);

            try
            {
                if (graph == null || !IsTraining)
                {
                    // Inference path - reuse pre-allocated output, use transposed weights for cache-friendly SIMD
                    if (_weightsTransposed == null)
                    {
                        RebuildTransposedWeights();
                    }

                    // Grow output buffer if current batch is larger than previous
                    if (batchSize > _inferenceOutputBatchSize)
                    {
                        _inferenceOutputNode?.Dispose();
                        _inferenceOutputNode = new AutogradNode(
                            new FastTensor<float>(batchSize, _outputSize, clearMemory: false), false);
                        _inferenceOutputBatchSize = batchSize;
                    }

                    var outSpan = _inferenceOutputNode.DataView.AsSpan().Slice(0, batchSize * _outputSize);
                    var inSpan = input.DataView.AsReadOnlySpan();
                    var wtSpan = _weightsTransposed.GetView().AsReadOnlySpan();
                    var bSpan = Biases.DataView.AsReadOnlySpan();

                    // Batched SIMD inference: each row treated independently
                    // Parallelize only for large batches - Parallel.For overhead (~50-100μs)
                    // exceeds benefit for small batches with lightweight models.
                    // Empirical threshold (Ryzen 9 9950X3D, 784->10 Linear):
                    //   batch*inputSize > 200k ~ batch >= 256 for InputSize=784.
                    //   Trade-off: small closure allocation (~6-7KB) for 3-4× speedup.
                    //   Below threshold: sequential SIMD wins due to zero dispatch overhead.
                    if ((long)batchSize * _inputSize > 200_000)
                    {
                        // Capture references outside lambda (ref struct Span cannot be captured)
                        var inputNode = input;
                        var weightsT = _weightsTransposed;
                        var biases = Biases;
                        var outputNode = _inferenceOutputNode;
                        var localInputSize = _inputSize;
                        var localOutputSize = _outputSize;

                        Parallel.For(0, batchSize, b =>
                        {
                            var localIn = inputNode.DataView.AsReadOnlySpan();
                            var localWt = weightsT.GetView().AsReadOnlySpan();
                            var localB = biases.DataView.AsReadOnlySpan();
                            var localOut = outputNode.DataView.AsSpan();

                            LinearInferenceSimd(
                                localIn.Slice(b * localInputSize, localInputSize),
                                localWt, localB,
                                localOut.Slice(b * localOutputSize, localOutputSize));
                        });
                    }
                    else
                    {
                        for (var b = 0; b < batchSize; b++)
                        {
                            LinearInferenceSimd(
                                inSpan.Slice(b * _inputSize, _inputSize),
                                wtSpan, bSpan,
                                outSpan.Slice(b * _outputSize, _outputSize));
                        }
                    }

                    return _inferenceOutputNode;
                }

                if (_weightsTransposed != null)
                {
                    _weightsTransposed.Dispose();
                    _weightsTransposed = null;
                }

                return TensorMath.Linear(graph, input, Weights, Biases);
            }
            finally
            {
                ModuleDiagnostics.End(ctx);
            }
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Weights;
            yield return Biases;
        }

        public void Save(BinaryWriter bw)
        {
            var wSpan = Weights.DataView.AsReadOnlySpan();
            for (var i = 0; i < wSpan.Length; i++)
            {
                bw.Write(wSpan[i]);
            }

            var bSpan = Biases.DataView.AsReadOnlySpan();
            for (var i = 0; i < bSpan.Length; i++)
            {
                bw.Write(bSpan[i]);
            }
        }

        public void Load(BinaryReader br)
        {
            var wSpan = Weights.DataView.AsSpan();
            for (var i = 0; i < wSpan.Length; i++)
            {
                wSpan[i] = br.ReadSingle();
            }

            var bSpan = Biases.DataView.AsSpan();
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

        private void RebuildTransposedWeights()
        {
            _weightsTransposed?.Dispose();
            _weightsTransposed = FastTensor<float>.FromView(Weights.DataView.Transpose2D());
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

            // Use TensorPrimitives.Dot for each output row.
            // Tuned implementation with AVX-512 support on modern CPUs.
            for (var j = 0; j < outputSize; j++)
            {
                var wRow = weightsT.Slice(j * inputSize, inputSize);
                output[j] = TensorPrimitives.Dot(input, wRow) + bias[j];
            }
        }
    }
}