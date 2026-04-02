using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Optimizers
{
    public sealed class Adam : IOptimizer, IDisposable
    {
        private readonly struct ParamState
        {
            public readonly AutogradNode Node;
            public readonly FastTensor<float> M;
            public readonly FastTensor<float> V;
            public readonly int Size;

            public ParamState(AutogradNode node)
            {
                Node = node;
                Size = node.Data.Size;

                // Używamy FastTensor z natywnym kształtem parametrów
                M = new FastTensor<float>(node.Data.Shape);
                M.AsSpan().Clear(); // Bezwzględnie czyścimy śmieci z ArrayPool

                V = new FastTensor<float>(node.Data.Shape);
                V.AsSpan().Clear();
            }
        }

        private readonly ParamState[] _states;

        // Używamy FastTensor jako potężnego, w 100% płaskiego bufora 1D!
        private readonly FastTensor<float> _bufGl2;
        private readonly FastTensor<float> _bufGSq;
        private readonly FastTensor<float> _bufMHat;
        private readonly FastTensor<float> _bufVHat;
        private readonly FastTensor<float> _bufUpd;

        public float LearningRate { get; set; }
        public float Beta1 { get; set; } = 0.9f;
        public float Beta2 { get; set; } = 0.999f;
        public float Epsilon { get; set; } = 1e-8f;
        public float WeightDecay { get; set; } = 0.0001f;

        private int _t = 0;

        public Adam(IEnumerable<AutogradNode> parameters, float learningRate = 0.001f)
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

            // Alokujemy globalne bufory 1D tylko pod największą warstwę
            _bufGl2 = new FastTensor<float>(maxSize);
            _bufGSq = new FastTensor<float>(maxSize);
            _bufMHat = new FastTensor<float>(maxSize);
            _bufVHat = new FastTensor<float>(maxSize);
            _bufUpd = new FastTensor<float>(maxSize);
        }

        public void Step()
        {
            _t++;

            var bc1 = 1f - MathF.Pow(Beta1, _t);
            var bc2 = 1f - MathF.Pow(Beta2, _t);
            var invBc1 = 1f / bc1;
            var invBc2 = 1f / bc2;

            foreach (var state in _states)
            {
                var p = state.Node;
                var n = state.Size;

                if (p.Grad == null) continue;

                var g = p.Grad.AsSpan();
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
                var b1Inv = 1f - Beta1;
                var b2Inv = 1f - Beta2;
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
                // Zabezpieczenie przed węzłami bez gradientu
                if (state.Node.Grad != null)
                {
                    state.Node.Grad.AsSpan().Clear();
                }
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