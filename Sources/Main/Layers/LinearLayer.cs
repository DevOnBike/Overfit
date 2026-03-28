namespace DevOnBike.Overfit.Layers
{
    /// <summary>
    /// W pełni połączona warstwa liniowa sieci neuronowej (Dense Layer).
    /// Wykonuje operację: Y = X * W + B
    /// </summary>
    public sealed class LinearLayer
    {
        public Tensor Weights { get; }
        public Tensor Biases { get; }

        public LinearLayer(int inputSize, int outputSize)
        {
            var wData = new FastMatrix<double>(inputSize, outputSize);
            var bData = new FastMatrix<double>(1, outputSize); // 1xD wektor

            InitializeWeights(wData.AsSpan(), inputSize);
            bData.AsSpan().Clear(); // Biasy na start mogą być zerami

            Weights = new Tensor(wData, requiresGrad: true);
            Biases = new Tensor(bData, requiresGrad: true);
        }

        /// <summary>
        /// Inicjalizacja Kaiming (He) - zapobiega eksplozji i zanikaniu gradientów.
        /// </summary>
        private void InitializeWeights(Span<double> span, int fanIn)
        {
            double stdDev = Math.Sqrt(2.0 / fanIn);
            for (int i = 0; i < span.Length; i++)
            {
                // Generowanie rozkładu normalnego (Box-Muller transform)
                double u1 = 1.0 - Random.Shared.NextDouble();
                double u2 = 1.0 - Random.Shared.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                
                span[i] = randStdNormal * stdDev;
            }
        }

        /// <summary>
        /// Przepływ sygnału przez warstwę: MatMul + Bias
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            var matMulResult = TensorMath.MatMul(input, Weights);
            return TensorMath.AddBias(matMulResult, Biases);
        }

        /// <summary>
        /// Zwraca parametry (wagi i biasy), by Optymalizator (SGD) mógł je zaktualizować.
        /// </summary>
        public IEnumerable<Tensor> Parameters()
        {
            yield return Weights;
            yield return Biases;
        }
    }
}