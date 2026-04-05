// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class LinearLayer : IModule
    {
        public AutogradNode Weights { get; private set; }
        public AutogradNode Biases { get; private set; }
        public bool IsTraining { get; private set; } = true;

        // Bufor dla inferencji (Zero-Allocation)
        private readonly AutogradNode _inferenceOutputNode;

        public LinearLayer(int inputSize, int outputSize)
        {
            var wData = new FastTensor<float>(inputSize, outputSize);
            // Inicjalizacja wag (He initialization)
            var stdDev = MathF.Sqrt(2f / inputSize);
            var wSpan = wData.AsSpan();
            for (var i = 0; i < wSpan.Length; i++) wSpan[i] = MathUtils.NextGaussian() * stdDev;

            Weights = new AutogradNode(wData, true);
            Biases = new AutogradNode(new FastTensor<float>(outputSize), true);

            // Inicjalizacja bufora wyjściowego
            var outData = new FastTensor<float>(1, outputSize);
            _inferenceOutputNode = new AutogradNode(outData, requiresGrad: false);
        }

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            // Jeśli graf jest null LUB jesteśmy w trybie Eval -> używamy optymalizacji
            if (graph == null || !IsTraining)
            {
                LinearInferenceSimd(input.Data, Weights.Data, Biases.Data, _inferenceOutputNode.Data);
                return _inferenceOutputNode;
            }

            // Tryb treningowy - standardowy Autograd
            return TensorMath.Linear(graph, input, Weights, Biases);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Weights;
            yield return Biases;
        }

        // --- Naprawa błędów Load/Save (zgodnie z nowym IModule) ---

        public void Save(BinaryWriter bw)
        {
            // Zapis surowych danych (kompatybilny z Pythonem)
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
            // Odczyt surowych danych
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
        }

        // --- Metody pomocnicze (nieinterfejsowe) ---

        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);
            Save(bw);
        }

        public void Load(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Brak pliku wag: {path}");
            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);
            Load(br);
        }

        // --- Silnik SIMD ---

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void LinearInferenceSimd(FastTensor<float> input, FastTensor<float> weights, FastTensor<float> bias, FastTensor<float> output)
        {
            var inSpan = input.AsReadOnlySpan();
            var wSpan = weights.AsReadOnlySpan();
            var bSpan = bias.AsReadOnlySpan();
            var outSpan = output.AsSpan();

            var inputSize = inSpan.Length;
            var outputSize = outSpan.Length;
            var vCount = Vector<float>.Count;

            for (var outIdx = 0; outIdx < outputSize; outIdx++)
            {
                var sum = bSpan[outIdx];
                var inIdx = 0;
                var wOffset = outIdx * inputSize;

                for (; inIdx <= inputSize - vCount; inIdx += vCount)
                {
                    var vIn = new Vector<float>(inSpan.Slice(inIdx));
                    var vW = new Vector<float>(wSpan.Slice(wOffset + inIdx));
                    sum += Vector.Dot(vIn, vW);
                }

                for (; inIdx < inputSize; inIdx++)
                {
                    sum += inSpan[inIdx] * wSpan[wOffset + inIdx];
                }

                outSpan[outIdx] = sum;
            }
        }

        public void Dispose()
        {
            Weights?.Dispose();
            Biases?.Dispose();
            _inferenceOutputNode?.Dispose();
        }
    }
}