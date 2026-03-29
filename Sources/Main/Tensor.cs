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
            var visited = new HashSet<Tensor>();    // Węzły wrzucone na stos
            var processed = new HashSet<Tensor>();  // Węzły w pełni przetworzone (dodane do topo)
            var stack = new Stack<Tensor>();

            stack.Push(this);
            visited.Add(this);

            // Iteracyjny Post-Order DFS
            while (stack.Count > 0)
            {
                var node = stack.Peek();
                bool allChildrenProcessed = true;

                foreach (var child in node._dependencies)
                {
                    if (!visited.Contains(child))
                    {
                        visited.Add(child);
                        stack.Push(child);
                        allChildrenProcessed = false;
                    }
                }

                // Gdy wszystkie dzieci zostały wrzucone na stos (lub już na nim są), 
                // zdejmujemy węzeł i dodajemy go do posortowanej listy
                if (allChildrenProcessed)
                {
                    stack.Pop();
                    
                    if (!processed.Contains(node))
                    {
                        processed.Add(node);
                        topo.Add(node);
                    }
                }
            }

            // Odwracamy listę i odpalamy propagację wstecz
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