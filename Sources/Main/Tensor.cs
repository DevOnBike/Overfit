using System;
using System.Collections.Generic;

namespace DevOnBike.Overfit
{
    /// <summary>
    /// Węzeł grafu obliczeniowego. Owija FastMatrix, przechowuje gradienty 
    /// oraz historię operacji do wstecznej propagacji (Autograd).
    /// </summary>
    public sealed class Tensor : IDisposable
    {
        public FastMatrix<double> Data { get; }
        public FastMatrix<double> Grad { get; }
        public bool RequiresGrad { get; }

        private readonly List<Tensor> _dependencies = [];
        private Action<Tensor> _backwardAction;

        private readonly bool _ownsData;

        public Tensor(FastMatrix<double> data, bool requiresGrad = true, bool ownsData = true)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            RequiresGrad = requiresGrad;
            Grad = new FastMatrix<double>(data.Rows, data.Cols);
            _ownsData = ownsData;
        }

        private Tensor(FastMatrix<double> data, List<Tensor> dependencies, Action<Tensor> backwardAction)
        {
            Data = data;
            RequiresGrad = true;
            Grad = new FastMatrix<double>(data.Rows, data.Cols);
            _dependencies = dependencies;
            _backwardAction = backwardAction;
            
            // Węzły tworzone przez TensorMath (np. wynik mnożenia) ZAWSZE są własnością tego Tensora
            _ownsData = true;
        }

        // ====================================================================
        // MAGIA AUTOGRADA - Wsteczna Propagacja
        // ====================================================================

        public void Backward()
        {
            var topo = new List<Tensor>();
            var visited = new HashSet<Tensor>();

            void BuildTopo(Tensor node)
            {
                if (!visited.Contains(node))
                {
                    visited.Add(node);
                    foreach (var child in node._dependencies)
                    {
                        BuildTopo(child);
                    }
                    topo.Add(node);
                }
            }

            BuildTopo(this);

            topo.Reverse();
            foreach (var node in topo)
            {
                node._backwardAction?.Invoke(node);
            }
        }

        public static Tensor CreateOperationResult(FastMatrix<double> data, List<Tensor> dependencies, Action<Tensor> backwardAction)
        {
            return new Tensor(data, dependencies, backwardAction);
        }

        public void Dispose()
        {
            // Sprzątamy Data TYLKO, gdy Tensor jest jej prawnym właścicielem
            if (_ownsData)
            {
#pragma warning disable IDISP007 // Don't dispose disposables you do not own
                Data?.Dispose();
#pragma warning restore IDISP007
            }
            
            Grad?.Dispose();
            _dependencies.Clear();
            _backwardAction = null;
        }
    }
}