namespace DevOnBike.Overfit.Optimizers
{
    public sealed class Adam
    {
        private readonly List<Tensor> _parameters;
        private readonly Dictionary<Tensor, FastMatrix<double>> _m = new(); 
        private readonly Dictionary<Tensor, FastMatrix<double>> _v = new(); 
        
        // Publiczne właściwości umożliwiające działanie LRScheduler
        public double LearningRate { get; set; }
        public double Beta1 { get; set; } = 0.9;
        public double Beta2 { get; set; } = 0.999;
        public double Epsilon { get; set; } = 1e-8;
        
        private int _t = 0; 

        public Adam(IEnumerable<Tensor> parameters, double learningRate = 0.001)
        {
            // Filtrujemy tylko parametry wymagające gradientu, by nie marnować cykli
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
            
            // Poprawki biasu (Bias Correction) obliczane raz na krok
            var bc1 = 1.0 - Math.Pow(Beta1, _t);
            var bc2 = 1.0 - Math.Pow(Beta2, _t);

            foreach (var p in _parameters)
            {
                // Bezpiecznik dla Janitora: upewniamy się, że wagi nie zostały zdisposowane
                if (p.Data == null || p.Grad == null) continue;

                var g = p.Grad.AsReadOnlySpan();
                var m = _m[p].AsSpan();
                var v = _v[p].AsSpan();
                var w = p.Data.AsSpan();

                // Cache stałych dla wydajniejszej pętli
                var lr = LearningRate;
                var b1 = Beta1;
                var b1Inv = 1.0 - Beta1;
                var b2 = Beta2;
                var b2Inv = 1.0 - Beta2;
                var eps = Epsilon;

                // Implementacja wzoru Adama:
                // m_t = b1 * m_{t-1} + (1-b1) * g_t
                // v_t = b2 * v_{t-1} + (1-b2) * g_t^2
                // w = w - lr * (m_t / bc1) / (sqrt(v_t / bc2) + eps)
                for (var i = 0; i < w.Length; i++)
                {
                    m[i] = b1 * m[i] + b1Inv * g[i];
                    v[i] = b2 * v[i] + b2Inv * g[i] * g[i];

                    var mHat = m[i] / bc1;
                    var vHat = v[i] / bc2;

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

        // Pozwala na restart treningu na tych samych wagach z nowym licznikiem biasu
        public void ResetTime() => _t = 0;
    }
}