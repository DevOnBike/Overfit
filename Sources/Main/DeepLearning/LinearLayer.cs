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
            Biases = new AutogradNode(new FastTensor<float>(1, outputSize), true);
        }

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(AutogradNode input) => TensorMath.Linear(input, Weights, Biases);

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
            bw.Write(Weights.Data.Shape[0]);
            bw.Write(Weights.Data.Shape[1]);
            foreach (var val in Weights.Data.AsSpan()) bw.Write(val);

            bw.Write(Biases.Data.Shape[0]);
            bw.Write(Biases.Data.Shape[1]);
            foreach (var val in Biases.Data.AsSpan()) bw.Write(val);
        }

        public void Load(BinaryReader br)
        {
            var wRows = br.ReadInt32();
            var wCols = br.ReadInt32();
            if (wRows != Weights.Data.Shape[0] || wCols != Weights.Data.Shape[1])
                throw new Exception("Wymiary wag w pliku nie pasują do architektury!");

            var wSpan = Weights.Data.AsSpan();
            for (var i = 0; i < wSpan.Length; i++) wSpan[i] = br.ReadSingle();

            var bRows = br.ReadInt32();
            var bCols = br.ReadInt32();
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