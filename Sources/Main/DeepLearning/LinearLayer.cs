using System.Numerics;
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
        private readonly AutogradNode _inferenceOutputNode;
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
                    if (batchSize != 1)
                    {
                        return TensorMath.Linear(null, input, Weights, Biases);
                    }

                    if (_weightsTransposed == null)
                    {
                        RebuildTransposedWeights();
                    }

                    LinearInferenceSimd(
                    input.DataView.AsReadOnlySpan(),
                    _weightsTransposed.GetView().AsReadOnlySpan(),
                    Biases.DataView.AsReadOnlySpan(),
                    _inferenceOutputNode.DataView.AsSpan());

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
