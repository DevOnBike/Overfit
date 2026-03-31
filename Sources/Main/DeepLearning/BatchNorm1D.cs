using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class BatchNorm1D : IModule
    {
        public AutogradNode Gamma { get; private set; }
        public AutogradNode Beta { get; private set; }

        public FastMatrix<double> RunningMean { get; private set; }
        public FastMatrix<double> RunningVar { get; private set; }

        public double Momentum { get; set; }
        public double Eps { get; set; }

        public bool IsTraining { get; private set; } = true;

        public BatchNorm1D(int features, double momentum = 0.1, double eps = 1e-5)
        {
            var gammaData = new FastMatrix<double>(1, features);
            gammaData.AsSpan().Fill(1.0);
            Gamma = new AutogradNode(gammaData, requiresGrad: true);

            var betaData = new FastMatrix<double>(1, features);
            betaData.AsSpan().Fill(0.0);
            Beta = new AutogradNode(betaData, requiresGrad: true);

            RunningMean = new FastMatrix<double>(1, features);
            RunningVar = new FastMatrix<double>(1, features);
            RunningVar.AsSpan().Fill(1.0);

            Momentum = momentum;
            Eps = eps;
        }

        public void Train()
        {
            IsTraining = true;
        }
        public void Eval()
        {
            IsTraining = false;
        }

        public AutogradNode Forward(AutogradNode input)
        {
            return TensorMath.BatchNorm1D(input, Gamma, Beta, RunningMean, RunningVar, Momentum, Eps, IsTraining);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Gamma;
            yield return Beta;
        }

        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);

            Save(bw);
        }

        public void Save(BinaryWriter bw)
        {
            foreach (var val in Gamma.Data.AsSpan()) bw.Write(val);
            foreach (var val in Beta.Data.AsSpan()) bw.Write(val);
            foreach (var val in RunningMean.AsSpan()) bw.Write(val);
            foreach (var val in RunningVar.AsSpan()) bw.Write(val);
        }

        public void Load(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Brak pliku wag: {path}");

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);

            Load(br);
        }

        public void Load(BinaryReader br)
        {
            var gSpan = Gamma.Data.AsSpan();
            for (var i = 0; i < gSpan.Length; i++) gSpan[i] = br.ReadDouble();

            var bSpan = Beta.Data.AsSpan();
            for (var i = 0; i < bSpan.Length; i++) bSpan[i] = br.ReadDouble();

            var rmSpan = RunningMean.AsSpan();
            for (var i = 0; i < rmSpan.Length; i++) rmSpan[i] = br.ReadDouble();

            var rvSpan = RunningVar.AsSpan();
            for (var i = 0; i < rvSpan.Length; i++) rvSpan[i] = br.ReadDouble();
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