namespace DevOnBike.Overfit.Optimizers
{
    public sealed class Adam
    {
        private readonly List<Tensor> _parameters;
        private readonly Dictionary<Tensor, FastMatrix<double>> _m = new(); // Pierwszy moment (średnia)
        private readonly Dictionary<Tensor, FastMatrix<double>> _v = new(); // Drugi moment (wariancja)
        
        public double LearningRate { get; set; }
        public double Beta1 { get; set; } = 0.9;
        public double Beta2 { get; set; } = 0.999;
        public double Epsilon { get; set; } = 1e-8;
        
        private int _t = 0; // Licznik kroków (czasu)

        public Adam(IEnumerable<Tensor> parameters, double learningRate = 0.001)
        {
            _parameters = new List<Tensor>(parameters);
            LearningRate = learningRate;

            foreach (var p in _parameters)
            {
                // Inicjalizujemy bufory pomocnicze zerami
                _m[p] = new FastMatrix<double>(p.Data.Rows, p.Data.Cols);
                _v[p] = new FastMatrix<double>(p.Data.Rows, p.Data.Cols);
            }
        }

        public void Step()
        {
            _t++; // Zwiększamy licznik epok/batchy
            
            // Współczynniki do poprawki biasu
            var biasCorrection1 = 1.0 - Math.Pow(Beta1, _t);
            var biasCorrection2 = 1.0 - Math.Pow(Beta2, _t);

            foreach (var p in _parameters)
            {
                if (!p.RequiresGrad)
                {
                    continue;
                }

                var g = p.Grad.AsReadOnlySpan();
                var m = _m[p].AsSpan();
                var v = _v[p].AsSpan();
                var w = p.Data.AsSpan();

                // Matematyka Adama w jednym przejściu (SIMD-friendly loop)
                // Używamy pętli ręcznej, bo TensorPrimitives nie ma "wszystkiego w jednym" 
                // dla złożonego wzoru Adama, a chcemy uniknąć alokacji tymczasowych.
                for (var i = 0; i < w.Length; i++)
                {
                    // 1. Aktualizacja momentów
                    m[i] = Beta1 * m[i] + (1.0 - Beta1) * g[i];
                    v[i] = Beta2 * v[i] + (1.0 - Beta2) * g[i] * g[i];

                    // 2. Poprawka biasu (na początku treningu m i v są sztucznie małe)
                    var mHat = m[i] / biasCorrection1;
                    var vHat = v[i] / biasCorrection2;

                    // 3. Aktualizacja wagi
                    w[i] -= LearningRate * mHat / (Math.Sqrt(vHat) + Epsilon);
                }
            }
        }

        public void ZeroGrad()
        {
            foreach (var p in _parameters)
            {
                p.Grad.Clear();
            }
        }
    }
}