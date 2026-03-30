using DevOnBike.Overfit.Core;
namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class ConvLayer
    {
        public AutogradNode Kernels { get; }
        private int _inC, _outC, _h, _w, _k;

        public ConvLayer(int inChannels, int outChannels, int h, int w, int kSize)
        {
            _inC = inChannels; _outC = outChannels; _h = h; _w = w; _k = kSize;
        
            var kData = new FastMatrix<double>(outChannels, inChannels * kSize * kSize);
            // Inicjalizacja Kaiming dla splotów
            InitializeKernels(kData.AsSpan(), inChannels * kSize * kSize);
        
            Kernels = new AutogradNode(kData, true);
        }

        private void InitializeKernels(Span<double> span, int fanIn)
        {
            var stdDev = Math.Sqrt(2.0 / fanIn);
            
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = (Random.Shared.NextDouble() * 2 - 1) * stdDev;
            }
        }

        public AutogradNode Forward(AutogradNode input)
        {
            return TensorMath.Conv2D(input, Kernels, _inC, _outC, _h, _w, _k);
        }

        // --- DODANE: Metoda wymagana przez Optymalizator ---
        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Kernels;
        }

        // --- DODANE: Metody zapisu i odczytu dla Beast Mode ---
        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);
            
            bw.Write(Kernels.Data.Rows);
            bw.Write(Kernels.Data.Cols);
            foreach (var val in Kernels.Data.AsSpan()) bw.Write(val);
        }

        public void Load(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Brak pliku filtrów: {path}");

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);

            var rows = br.ReadInt32();
            var cols = br.ReadInt32();

            if (rows != Kernels.Data.Rows || cols != Kernels.Data.Cols)
                throw new Exception("Wymiary filtrów w pliku nie pasują do architektury ConvLayer!");

            var span = Kernels.Data.AsSpan();
            for (var i = 0; i < span.Length; i++) span[i] = br.ReadDouble();
        }
    }
}