// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.DeepLearning.Diagnostics;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core; // Wpinamy Core

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class LinearLayer : IModule
    {
        private AutogradNode _inferenceOutputNode;
        private int _inferenceOutputBatchSize = 1;
        private readonly int _inputSize;
        private readonly int _outputSize;

        // Zmieniono FastTensor na TensorStorage
        private TensorStorage<float>? _weightsTransposed;

        public LinearLayer(int inputSize, int outputSize)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;

            var wData = new TensorStorage<float>(inputSize * outputSize, clearMemory: false);
            var stdDev = MathF.Sqrt(2f / inputSize);
            var wSpan = wData.AsSpan();

            for (var i = 0; i < wSpan.Length; i++)
            {
                wSpan[i] = MathUtils.NextGaussian() * stdDev;
            }

            // Rozdzielony Kształt i Magazyn!
            Weights = new AutogradNode(wData, new TensorShape(inputSize, outputSize), requiresGrad: true);

            var bData = new TensorStorage<float>(outputSize, clearMemory: true);
            Bias = new AutogradNode(bData, new TensorShape(outputSize), requiresGrad: true);
        }

        public AutogradNode Weights { get; }
        public AutogradNode Bias { get; }
        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var ctx = ModuleDiagnostics.Begin(nameof(LinearLayer), "forward_train", true, input.Shape.D0, _inputSize, input.Shape.D0, _outputSize);
            try
            {
                return TensorMath.Linear(graph, input, Weights, Bias);
            }
            finally
            {
                ModuleDiagnostics.End(ctx);
            }
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Weights;
            yield return Bias;
        }

        public void InvalidateParameterCaches()
        {
            if (!IsTraining) RebuildTransposedWeights();
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(Weights.DataView.Size);
            foreach (var val in Weights.DataView.AsReadOnlySpan()) bw.Write(val);

            bw.Write(Bias.DataView.Size);
            foreach (var val in Bias.DataView.AsReadOnlySpan()) bw.Write(val);
        }

        public void Load(BinaryReader br)
        {
            var lenW = br.ReadInt32();
            if (lenW != Weights.DataView.Size) throw new Exception("Weights mismatch");

            var wSpan = Weights.DataView.AsSpan();
            for (var i = 0; i < lenW; i++) wSpan[i] = br.ReadSingle();

            var lenB = br.ReadInt32();
            if (lenB != Bias.DataView.Size) throw new Exception("Bias mismatch");

            var bSpan = Bias.DataView.AsSpan();
            for (var i = 0; i < lenB; i++) bSpan[i] = br.ReadSingle();
        }

        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);
            Save(bw);
        }

        public void Load(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Weight file not found: {path}");

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);
            Load(br);
        }

        private void RebuildTransposedWeights()
        {
            _weightsTransposed?.Dispose();
            // FABRYKA: Zamiast FastTensor.FromView
            _weightsTransposed = TensorFactory.Materialize(Weights.DataView.Transpose2D());
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
            if (_weightsTransposed == null) RebuildTransposedWeights();

            var batchSize = input.Length / _inputSize;

            if (batchSize == 1)
            {
                LinearInferenceSimd(input, _weightsTransposed.AsReadOnlySpan(), Bias.DataView.AsReadOnlySpan(), output);
            }
            else
            {
                var wTSpan = _weightsTransposed.AsReadOnlySpan();
                var bSpan = Bias.DataView.AsReadOnlySpan();

                for (var b = 0; b < batchSize; b++)
                {
                    var inSlice = input.Slice(b * _inputSize, _inputSize);
                    var outSlice = output.Slice(b * _outputSize, _outputSize);
                    LinearInferenceSimd(inSlice, wTSpan, bSpan, outSlice);
                }
            }
        }

        public void Dispose()
        {
            Weights?.Dispose();
            Bias?.Dispose();
            _weightsTransposed?.Dispose();
            _inferenceOutputNode?.Dispose();
        }
    }
}