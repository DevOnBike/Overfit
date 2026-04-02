using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class ConvLayer : IModule
    {
        public AutogradNode Kernels { get; }
        private int _inC, _outC, _h, _w, _k;

        public bool IsTraining { get; private set; } = true;

        public ConvLayer(int inChannels, int outChannels, int h, int w, int kSize)
        {
            _inC = inChannels; _outC = outChannels; _h = h; _w = w; _k = kSize;

            var kData = new FastMatrix<float>(outChannels, inChannels * kSize * kSize);

            InitializeKernels(kData.AsSpan(), inChannels * kSize * kSize);

            Kernels = new AutogradNode(kData, true);
        }

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        private void InitializeKernels(Span<float> span, int fanIn)
        {
            var stdDev = MathF.Sqrt(2f / fanIn);

            for (var i = 0; i < span.Length; i++)
            {
                span[i] = MathUtils.NextGaussian() * stdDev;
            }
        }

        public AutogradNode Forward(AutogradNode input)
        {
            return TensorMath.Conv2D(input, Kernels, _inC, _outC, _h, _w, _k);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Kernels;
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(Kernels.Data.Rows);
            bw.Write(Kernels.Data.Cols);

            foreach (var val in Kernels.Data.AsSpan()) bw.Write(val);
        }

        public void Load(BinaryReader br)
        {
            var rows = br.ReadInt32();
            var cols = br.ReadInt32();
            if (rows != Kernels.Data.Rows || cols != Kernels.Data.Cols)
                throw new Exception("Wymiary filtrów w pliku nie pasują do architektury ConvLayer!");

            var span = Kernels.Data.AsSpan();
            for (var i = 0; i < span.Length; i++) span[i] = br.ReadSingle();
        }

        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);

            Save(bw);
        }

        public void Load(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Brak pliku filtrów: {path}");
            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);

            Load(br);
        }

        public void Dispose()
        {
            Kernels?.Dispose();
        }
    }
}