namespace DevOnBike.Overfit.Layers
{
    public sealed class ConvLayer
    {
        public Tensor Kernels { get; }
        private int _inC, _outC, _h, _w, _k;

        public ConvLayer(int inChannels, int outChannels, int h, int w, int kSize)
        {
            _inC = inChannels; _outC = outChannels; _h = h; _w = w; _k = kSize;
        
            var kData = new FastMatrix<double>(outChannels, inChannels * kSize * kSize);
            // Inicjalizacja Kaiming dla splotów
            InitializeKernels(kData.AsSpan(), inChannels * kSize * kSize);
        
            Kernels = new Tensor(kData, true);
        }

        private void InitializeKernels(Span<double> span, int fanIn)
        {
            var stdDev = Math.Sqrt(2.0 / fanIn);
            
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = (Random.Shared.NextDouble() * 2 - 1) * stdDev;
            }
        }

        public Tensor Forward(Tensor input)
        {
            return TensorMath.Conv2D(input, Kernels, _inC, _outC, _h, _w, _k);
        }
    }
}