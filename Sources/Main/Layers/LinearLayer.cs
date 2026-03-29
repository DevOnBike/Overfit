namespace DevOnBike.Overfit.Layers
{
    public class LinearLayer
    {
        public Tensor Weights { get; private set; }
        public Tensor Biases { get; private set; } // Nazwa zgodna z Twoim Dispose()

        public LinearLayer(int inputSize, int outputSize)
        {
            var wData = new FastMatrix<double>(inputSize, outputSize);
            var stdDev = Math.Sqrt(2.0 / inputSize);
            for (var i = 0; i < wData.AsSpan().Length; i++)
            {
                wData.AsSpan()[i] = (Random.Shared.NextDouble() * 2 - 1) * stdDev;
            }

            Weights = new Tensor(wData, true);
            Biases = new Tensor(new FastMatrix<double>(1, outputSize), true);
        }

        public Tensor Forward(Tensor input)
        {
            // Jedno wywołanie, jeden Tensor, zero wycieków
            return TensorMath.Linear(input, Weights, Biases);
        }

        // --- METODY PROSTEGO ZAPISU I ODCZYTU (FIX CS1061) ---

        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);
            Save(bw); // Wywołuje wersję niskopoziomową
        }

        public void Load(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Brak pliku wag: {path}");

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);
            Load(br); // Wywołuje wersję niskopoziomową
        }

        // --- WERSJE NISKOPOZIOMOWE (Dla BinaryReader/Writer) ---

        public void Save(BinaryWriter bw)
        {
            // Wagi
            bw.Write(Weights.Data.Rows);
            bw.Write(Weights.Data.Cols);
            foreach (var val in Weights.Data.AsSpan()) bw.Write(val);

            // Biasy
            bw.Write(Biases.Data.Rows);
            bw.Write(Biases.Data.Cols);
            foreach (var val in Biases.Data.AsSpan()) bw.Write(val);
        }

        public void Load(BinaryReader br)
        {
            // Odczyt wag (z prostą weryfikacją wymiarów)
            var wRows = br.ReadInt32();
            var wCols = br.ReadInt32();
            if (wRows != Weights.Data.Rows || wCols != Weights.Data.Cols)
                throw new Exception("Wymiary wag w pliku nie pasują do architektury!");

            var wSpan = Weights.Data.AsSpan();
            for (var i = 0; i < wSpan.Length; i++) wSpan[i] = br.ReadDouble();

            // Odczyt biasów
            var bRows = br.ReadInt32();
            var bCols = br.ReadInt32();
            var bSpan = Biases.Data.AsSpan();
            for (var i = 0; i < bSpan.Length; i++) bSpan[i] = br.ReadDouble();
        }

        public IEnumerable<Tensor> Parameters()
        {
            yield return Weights;
            yield return Biases;
        }
    }
}