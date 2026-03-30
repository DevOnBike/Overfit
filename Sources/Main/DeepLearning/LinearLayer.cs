using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class LinearLayer : IDisposable // Zmiana: dodano 'sealed'
    {
        public AutogradNode Weights { get; private set; } //[cite: 2]
        public AutogradNode Biases { get; private set; }  //[cite: 2]

        public LinearLayer(int inputSize, int outputSize) //[cite: 2]
        {
            var wData = new FastMatrix<double>(inputSize, outputSize); //[cite: 2]
            var stdDev = Math.Sqrt(2.0 / inputSize); // Kaiming He Fan-In[cite: 2]

            var wSpan = wData.AsSpan();
            for (var i = 0; i < wSpan.Length; i++)
            {
                // Zmiana: Zastosowanie poprawnego rozkładu normalnego Kaiminga za pomocą Box-Mullera
                wSpan[i] = MathUtils.NextGaussian() * stdDev;
            }

            Weights = new AutogradNode(wData, true); //[cite: 2]
            Biases = new AutogradNode(new FastMatrix<double>(1, outputSize), true); //[cite: 2]
        }

        public AutogradNode Forward(AutogradNode input) //[cite: 2]
        {
            return TensorMath.Linear(input, Weights, Biases); //[cite: 2]
        }

        // --- METODY PROSTEGO ZAPISU I ODCZYTU --- //[cite: 2]

        public void Save(string path) //[cite: 2]
        {
            using var fs = new FileStream(path, FileMode.Create); //[cite: 2]
            using var bw = new BinaryWriter(fs); //[cite: 2]
            Save(bw); //[cite: 2]
        }

        public void Load(string path) //[cite: 2]
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Brak pliku wag: {path}"); //[cite: 2]

            using var fs = new FileStream(path, FileMode.Open); //[cite: 2]
            using var br = new BinaryReader(fs); //[cite: 2]
            Load(br); //[cite: 2]
        }

        // --- WERSJE NISKOPOZIOMOWE --- //[cite: 2]

        public void Save(BinaryWriter bw) //[cite: 2]
        {
            bw.Write(Weights.Data.Rows); //[cite: 2]
            bw.Write(Weights.Data.Cols); //[cite: 2]
            foreach (var val in Weights.Data.AsSpan()) bw.Write(val); //[cite: 2]

            bw.Write(Biases.Data.Rows); //[cite: 2]
            bw.Write(Biases.Data.Cols); //[cite: 2]
            foreach (var val in Biases.Data.AsSpan()) bw.Write(val); //[cite: 2]
        }

        public void Load(BinaryReader br) //[cite: 2]
        {
            var wRows = br.ReadInt32(); //[cite: 2]
            var wCols = br.ReadInt32(); //[cite: 2]
            if (wRows != Weights.Data.Rows || wCols != Weights.Data.Cols) //[cite: 2]
                throw new Exception("Wymiary wag w pliku nie pasują do architektury!"); //[cite: 2]

            var wSpan = Weights.Data.AsSpan(); //[cite: 2]
            for (var i = 0; i < wSpan.Length; i++) wSpan[i] = br.ReadDouble(); //[cite: 2]

            var bRows = br.ReadInt32(); //[cite: 2]
            var bCols = br.ReadInt32(); //[cite: 2]
            var bSpan = Biases.Data.AsSpan(); //[cite: 2]
            for (var i = 0; i < bSpan.Length; i++) bSpan[i] = br.ReadDouble(); //[cite: 2]
        }

        public IEnumerable<AutogradNode> Parameters() //[cite: 2]
        {
            yield return Weights; //[cite: 2]
            yield return Biases; //[cite: 2]
        }

        public void Dispose() //[cite: 2]
        {
            Weights?.Dispose(); //[cite: 2]
            Biases?.Dispose(); //[cite: 2]
        }
    }
}