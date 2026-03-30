using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class ConvLayer : IDisposable // Zmiana: Dodanie IDisposable[cite: 3]
    {
        public AutogradNode Kernels { get; } //[cite: 3]
        private int _inC, _outC, _h, _w, _k; //[cite: 3]

        public ConvLayer(int inChannels, int outChannels, int h, int w, int kSize) //[cite: 3]
        {
            _inC = inChannels; _outC = outChannels; _h = h; _w = w; _k = kSize; //[cite: 3]

            var kData = new FastMatrix<double>(outChannels, inChannels * kSize * kSize); //[cite: 3]
            InitializeKernels(kData.AsSpan(), inChannels * kSize * kSize); //[cite: 3]

            Kernels = new AutogradNode(kData, true); //[cite: 3]
        }

        private void InitializeKernels(Span<double> span, int fanIn) //[cite: 3]
        {
            var stdDev = Math.Sqrt(2.0 / fanIn); //[cite: 3]

            for (var i = 0; i < span.Length; i++) //[cite: 3]
            {
                // Zmiana: Wykorzystanie poprawnego rozkładu normalnego Kaiminga z Box-Mullerem
                span[i] = MathUtils.NextGaussian() * stdDev;
            }
        }

        public AutogradNode Forward(AutogradNode input) //[cite: 3]
        {
            return TensorMath.Conv2D(input, Kernels, _inC, _outC, _h, _w, _k); //[cite: 3]
        }

        public IEnumerable<AutogradNode> Parameters() //[cite: 3]
        {
            yield return Kernels; //[cite: 3]
        }

        // --- Zapis i odczyt (Beast Mode) --- //[cite: 3]
        public void Save(string path) //[cite: 3]
        {
            using var fs = new FileStream(path, FileMode.Create); //[cite: 3]
            using var bw = new BinaryWriter(fs); //[cite: 3]

            bw.Write(Kernels.Data.Rows); //[cite: 3]
            bw.Write(Kernels.Data.Cols); //[cite: 3]
            foreach (var val in Kernels.Data.AsSpan()) bw.Write(val); //[cite: 3]
        }

        public void Load(string path) //[cite: 3]
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Brak pliku filtrów: {path}"); //[cite: 3]

            using var fs = new FileStream(path, FileMode.Open); //[cite: 3]
            using var br = new BinaryReader(fs); //[cite: 3]

            var rows = br.ReadInt32(); //[cite: 3]
            var cols = br.ReadInt32(); //[cite: 3]

            if (rows != Kernels.Data.Rows || cols != Kernels.Data.Cols) //[cite: 3]
                throw new Exception("Wymiary filtrów w pliku nie pasują do architektury ConvLayer!"); //[cite: 3]

            var span = Kernels.Data.AsSpan(); //[cite: 3]
            for (var i = 0; i < span.Length; i++) span[i] = br.ReadDouble(); //[cite: 3]
        }

        // Zmiana: Implementacja poprawnego zwalniania zasobów.
        public void Dispose()
        {
            Kernels?.Dispose();
        }
    }
}