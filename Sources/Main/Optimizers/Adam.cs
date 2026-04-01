using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Optimizers
{
    public sealed class Adam : IOptimizer, IDisposable
    {
        private readonly struct ParamState
        {
            public readonly AutogradNode Node;
            public readonly FastMatrix<double> M;
            public readonly FastMatrix<double> V;
            public readonly int Size;

            public ParamState(AutogradNode node)
            {
                Node = node;
                Size = node.Data.Rows * node.Data.Cols;
                M = new FastMatrix<double>(node.Data.Rows, node.Data.Cols);
                V = new FastMatrix<double>(node.Data.Rows, node.Data.Cols);
            }
        }

        private readonly ParamState[] _states;

        // Pojedyncze bufory wielokrotnego użytku! Zero ThreadLocal, zero locków.
        private readonly FastBuffer<double> _bufGl2;
        private readonly FastBuffer<double> _bufGSq;
        private readonly FastBuffer<double> _bufMHat;
        private readonly FastBuffer<double> _bufVHat;
        private readonly FastBuffer<double> _bufUpd;

        public double LearningRate { get; set; }
        public double Beta1 { get; set; } = 0.9;
        public double Beta2 { get; set; } = 0.999;
        public double Epsilon { get; set; } = 1e-8;
        public double WeightDecay { get; set; } = 0.0001;

        private int _t = 0;

        public Adam(IEnumerable<AutogradNode> parameters, double learningRate = 0.001)
        {
            LearningRate = learningRate;

            var paramList = parameters.Where(p => p.RequiresGrad).ToList();
            _states = new ParamState[paramList.Count];

            var maxSize = 1;

            for (var i = 0; i < paramList.Count; i++)
            {
                _states[i] = new ParamState(paramList[i]);
                if (_states[i].Size > maxSize) maxSize = _states[i].Size;
            }

            // Alokujemy globalne bufory tylko pod największą warstwę
            _bufGl2 = new FastBuffer<double>(maxSize);
            _bufGSq = new FastBuffer<double>(maxSize);
            _bufMHat = new FastBuffer<double>(maxSize);
            _bufVHat = new FastBuffer<double>(maxSize);
            _bufUpd = new FastBuffer<double>(maxSize);
        }

        public void Step()
        {
            _t++;

            var bc1 = 1.0 - Math.Pow(Beta1, _t);
            var bc2 = 1.0 - Math.Pow(Beta2, _t);
            var invBc1 = 1.0 / bc1;
            var invBc2 = 1.0 / bc2;

            // Płaska iteracja zamiast Parallel.ForEach
            foreach (var state in _states)
            {
                var p = state.Node;
                var n = state.Size;
                var g = p.Grad.AsReadOnlySpan();
                var m = state.M.AsSpan();
                var v = state.V.AsSpan();
                var w = p.Data.AsSpan();

                // Wycinamy bezpiecznie potrzebny fragment pre-alokowanego bufora
                var gl2 = _bufGl2.AsSpan()[..n];
                var gSq = _bufGSq.AsSpan()[..n];
                var mHat = _bufMHat.AsSpan()[..n];
                var vHat = _bufVHat.AsSpan()[..n];
                var upd = _bufUpd.AsSpan()[..n];

                var wd = WeightDecay;
                var b1 = Beta1;
                var b2 = Beta2;
                var b1Inv = 1.0 - Beta1;
                var b2Inv = 1.0 - Beta2;
                var eps = Epsilon;
                var lr = LearningRate;

                // --- 1. gWithL2 = g + wd*w ---
                TensorPrimitives.MultiplyAdd(w, wd, g, gl2);

                // --- 2. m = b1*m + b1Inv*gWithL2 ---
                TensorPrimitives.Multiply(m, b1, mHat);
                TensorPrimitives.MultiplyAdd(gl2, b1Inv, mHat, m);

                // --- 3. v = b2*v + b2Inv*gWithL2² ---
                TensorPrimitives.Multiply(gl2, gl2, gSq);
                TensorPrimitives.Multiply(v, b2, vHat);
                TensorPrimitives.MultiplyAdd(gSq, b2Inv, vHat, v);

                // --- 4. mHat = m / bc1 ---
                TensorPrimitives.Multiply(m, invBc1, mHat);

                // --- 5. vHat = sqrt(v/bc2) + eps ---
                TensorPrimitives.Multiply(v, invBc2, vHat);
                TensorPrimitives.Sqrt(vHat, vHat);
                TensorPrimitives.Add(vHat, eps, vHat);

                // --- 6. w -= lr * mHat / vHat ---
                TensorPrimitives.Divide(mHat, vHat, upd);
                TensorPrimitives.MultiplyAdd(upd, -lr, w, w);
            }
        }

        public void ZeroGrad()
        {
            foreach (var state in _states)
            {
                state.Node.Grad.Clear();
            }
        }

        public void ResetTime()
        {
            _t = 0;
        }

        public void Dispose()
        {
            foreach (var state in _states)
            {
                state.M?.Dispose();
                state.V?.Dispose();
            }

            _bufGl2.Dispose();
            _bufGSq.Dispose();
            _bufMHat.Dispose();
            _bufVHat.Dispose();
            _bufUpd.Dispose();
        }
    }
}