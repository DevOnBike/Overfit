namespace DevOnBike.Overfit.Layers
{
    public class BatchNorm1D
    {
        public Tensor Gamma { get; private set; }
        public Tensor Beta { get; private set; }
        
        // Statystyki do użycia w Inference
        public FastMatrix<double> RunningMean { get; private set; }
        public FastMatrix<double> RunningVar { get; private set; }
        
        public double Momentum { get; set; }
        public double Eps { get; set; }

        public BatchNorm1D(int features, double momentum = 0.1, double eps = 1e-5)
        {
            // Gamma inicjalizujemy na 1.0 (domyślna wariancja)
            var gammaData = new FastMatrix<double>(1, features);
            gammaData.AsSpan().Fill(1.0);
            Gamma = new Tensor(gammaData, requiresGrad: true);

            // Beta inicjalizujemy na 0.0 (domyślna średnia)
            var betaData = new FastMatrix<double>(1, features);
            betaData.AsSpan().Fill(0.0);
            Beta = new Tensor(betaData, requiresGrad: true);

            RunningMean = new FastMatrix<double>(1, features);
            RunningVar = new FastMatrix<double>(1, features);
            RunningVar.AsSpan().Fill(1.0);

            Momentum = momentum;
            Eps = eps;
        }

        public Tensor Forward(Tensor input, bool isTraining)
        {
            return TensorMath.BatchNorm1D(input, Gamma, Beta, RunningMean, RunningVar, Momentum, Eps, isTraining);
        }

        public IEnumerable<Tensor> Parameters()
        {
            yield return Gamma;
            yield return Beta;
        }

        public void Save(string path)
        {
            using var fs = new System.IO.FileStream(path, System.IO.FileMode.Create);
            using var bw = new System.IO.BinaryWriter(fs);
            
            foreach (var val in Gamma.Data.AsSpan()) bw.Write(val);
            foreach (var val in Beta.Data.AsSpan()) bw.Write(val);
            foreach (var val in RunningMean.AsSpan()) bw.Write(val);
            foreach (var val in RunningVar.AsSpan()) bw.Write(val);
        }

        public void Load(string path)
        {
            using var fs = new System.IO.FileStream(path, System.IO.FileMode.Open);
            using var br = new System.IO.BinaryReader(fs);
            
            var gSpan = Gamma.Data.AsSpan();
            for (int i = 0; i < gSpan.Length; i++) gSpan[i] = br.ReadDouble();
            
            var bSpan = Beta.Data.AsSpan();
            for (int i = 0; i < bSpan.Length; i++) bSpan[i] = br.ReadDouble();
            
            var rmSpan = RunningMean.AsSpan();
            for (int i = 0; i < rmSpan.Length; i++) rmSpan[i] = br.ReadDouble();
            
            var rvSpan = RunningVar.AsSpan();
            for (int i = 0; i < rvSpan.Length; i++) rvSpan[i] = br.ReadDouble();
        }
    }
}