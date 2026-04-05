// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class LinearLayer : IModule
    {
        public AutogradNode Weights { get; private set; }
        public AutogradNode Biases { get; private set; }
        public bool IsTraining { get; private set; } = true;

        public LinearLayer(int inputSize, int outputSize)
        {
            var wData = new FastTensor<float>(inputSize, outputSize);
            var stdDev = MathF.Sqrt(2f / inputSize);
            var wSpan = wData.AsSpan();

            for (var i = 0; i < wSpan.Length; i++) wSpan[i] = MathUtils.NextGaussian() * stdDev;

            Weights = new AutogradNode(wData, true);
            Biases = new AutogradNode(new FastTensor<float>(outputSize), true);
        }

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.Linear(graph, input, Weights, Biases);
        }

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

        public void Save(BinaryWriter bw)
        {
            // Zapisujemy wyłącznie surowe liczby (bez nagłówków wymiarów), 
            // zachowując kompatybilność ze skryptem Pythonowym.
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
            // Czytamy płaski strumień floatów (najpierw wagi, potem bias)
            // Zakładamy, że plik .bin ma dokładnie taką samą strukturę jak architektura sieci.

            var wSpan = Weights.Data.AsSpan();
            for (var i = 0; i < wSpan.Length; i++) wSpan[i] = br.ReadSingle();

            var bSpan = Biases.Data.AsSpan();
            for (var i = 0; i < bSpan.Length; i++) bSpan[i] = br.ReadSingle();
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Weights;
            yield return Biases;
        }

        public void Dispose()
        {
            Weights?.Dispose();
            Biases?.Dispose();
        }
    }
}