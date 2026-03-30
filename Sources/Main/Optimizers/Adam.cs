using DevOnBike.Overfit.Core;
namespace DevOnBike.Overfit.Optimizers
{
    public sealed class Adam
    {
        private readonly List<AutogradNode> _parameters;
        private readonly Dictionary<AutogradNode, FastMatrix<double>> _m = new();
        private readonly Dictionary<AutogradNode, FastMatrix<double>> _v = new();

        public double LearningRate { get; set; }
        public double Beta1 { get; set; } = 0.9;
        public double Beta2 { get; set; } = 0.999;
        public double Epsilon { get; set; } = 1e-8;

        // NOWOŚĆ: Parametr kary L2 (Weight Decay)
        public double WeightDecay { get; set; } = 0.0001;

        private int _t = 0;

        public Adam(IEnumerable<AutogradNode> parameters, double learningRate = 0.001)
        {
            _parameters = parameters.Where(p => p.RequiresGrad).ToList();
            LearningRate = learningRate;

            foreach (var p in _parameters)
            {
                _m[p] = new FastMatrix<double>(p.Data.Rows, p.Data.Cols);
                _v[p] = new FastMatrix<double>(p.Data.Rows, p.Data.Cols);
            }
        }

        public void Step()
        {
            _t++;

            // Poprawki biasu obliczane raz na krok optymalizatora
            var bc1 = 1.0 - Math.Pow(Beta1, _t);
            var bc2 = 1.0 - Math.Pow(Beta2, _t);

            foreach (var p in _parameters)
            {
                // Bezpiecznik dla Janitora
                if (p.Data == null || p.Grad == null) continue;

                var g = p.Grad.AsReadOnlySpan();
                var m = _m[p].AsSpan();
                var v = _v[p].AsSpan();
                var w = p.Data.AsSpan();

                // Cache stałych dla wydajności pętli[cite: 5]
                var lr = LearningRate;
                var wd = WeightDecay;
                var b1 = Beta1;
                var b1Inv = 1.0 - Beta1;
                var b2 = Beta2;
                var b2Inv = 1.0 - Beta2;
                var eps = Epsilon;

                for (var i = 0; i < w.Length; i++)
                {
                    // --- L2 Regularization ---
                    // Dodajemy pochodną kary (wd * w) do aktualnego gradientu
                    var gWithL2 = g[i] + wd * w[i];

                    // 1. Aktualizacja momentów z uwzględnieniem L2[cite: 5]
                    m[i] = b1 * m[i] + b1Inv * gWithL2;
                    v[i] = b2 * v[i] + b2Inv * gWithL2 * gWithL2;

                    // 2. Korekta biasu[cite: 5]
                    var mHat = m[i] / bc1;
                    var vHat = v[i] / bc2;

                    // 3. Aktualizacja wagi[cite: 5]
                    w[i] -= lr * mHat / (Math.Sqrt(vHat) + eps);
                }
            }
        }

        public void ZeroGrad()
        {
            foreach (var p in _parameters)
            {
                p.Grad?.Clear();
            }
        }

        public void ResetTime() => _t = 0;
    }
}