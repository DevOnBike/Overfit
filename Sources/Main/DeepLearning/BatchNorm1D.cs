// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class BatchNorm1D : IModule
    {
        public AutogradNode Gamma { get; private set; }
        public AutogradNode Beta { get; private set; }
        public FastTensor<float> RunningMean { get; private set; }
        public FastTensor<float> RunningVar { get; private set; }

        public bool IsTraining { get; private set; } = true;
        public float Momentum { get; set; } = 0.1f;
        public float Eps { get; set; } = 1e-5f;

        public BatchNorm1D(int numFeatures)
        {
            // KRYTYCZNA POPRAWKA: Używamy tensorów 1D (numFeatures), a nie 2D (1, numFeatures)
            Gamma = new AutogradNode(new FastTensor<float>(numFeatures), true);
            Beta = new AutogradNode(new FastTensor<float>(numFeatures), true);

            Gamma.Data.AsSpan().Fill(1f); // Domyślnie mnożnik to 1
            Beta.Data.AsSpan().Fill(0f);  // Domyślnie przesunięcie to 0

            RunningMean = new FastTensor<float>(numFeatures);
            RunningVar = new FastTensor<float>(numFeatures);

            RunningMean.AsSpan().Fill(0f);
            RunningVar.AsSpan().Fill(1f); // Domyślna wariancja to 1
        }

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.BatchNorm1D(graph, input, Gamma, Beta, RunningMean, RunningVar, Momentum, Eps, IsTraining);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Gamma;
            yield return Beta;
        }

        public void Save(BinaryWriter bw)
        {
            var len = Gamma.Data.Size;
            bw.Write(len);

            foreach (var val in Gamma.Data.AsSpan()) bw.Write(val);
            foreach (var val in Beta.Data.AsSpan()) bw.Write(val);
            foreach (var val in RunningMean.AsSpan()) bw.Write(val);
            foreach (var val in RunningVar.AsSpan()) bw.Write(val);
        }

        public void Load(BinaryReader br)
        {
            var len = br.ReadInt32();
            if (len != Gamma.Data.Size)
                throw new Exception($"Wymiary wag w pliku ({len}) nie pasują do architektury ({Gamma.Data.Size})!");

            var gSpan = Gamma.Data.AsSpan();
            for (var i = 0; i < len; i++) gSpan[i] = br.ReadSingle();

            var bSpan = Beta.Data.AsSpan();
            for (var i = 0; i < len; i++) bSpan[i] = br.ReadSingle();

            var rmSpan = RunningMean.AsSpan();
            for (var i = 0; i < len; i++) rmSpan[i] = br.ReadSingle();

            var rvSpan = RunningVar.AsSpan();
            for (var i = 0; i < len; i++) rvSpan[i] = br.ReadSingle();
        }

        public void Dispose()
        {
            Gamma?.Dispose();
            Beta?.Dispose();
            RunningMean?.Dispose();
            RunningVar?.Dispose();
        }
    }
}