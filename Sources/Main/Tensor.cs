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

        internal readonly List<Tensor> _dependencies = [];
        internal Action<Tensor> _backwardAction;

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

        public List<Tensor> Backward()
        {
            // 1. BUDOWA GRAFU OBLICZENIOWEGO (Sortowanie Topologiczne)
            // Musimy przetwarzać węzły od końca (Loss) do początku (Inputs)
            var topo = new List<Tensor>();
            var visited = new HashSet<Tensor>();

            void BuildTopo(Tensor node)
            {
                if (!visited.Contains(node))
                {
                    visited.Add(node);
                    foreach (var dep in node._dependencies)
                    {
                        BuildTopo(dep);
                    }
                    topo.Add(node);
                }
            }

            BuildTopo(this);
            topo.Reverse(); // Odwracamy, aby zacząć od 'this' (zazwyczaj Loss)

            // 2. INTELIGENTNY START GRADIENTU
            // Reguła łańcuchowa (Chain Rule) wymaga punktu startowego: dLoss/dLoss = 1.0
            // Sprawdzamy, czy ktoś już ręcznie nie wpisał gradientu (np. w testach)
            var isGradZero = true;
            var gradSpan = Grad.AsSpan();
            for (var i = 0; i < gradSpan.Length; i++)
            {
                if (gradSpan[i] != 0)
                {
                    isGradZero = false;
                    break;
                }
            }

            if (isGradZero)
            {
                // Jeśli gradient jest pusty, inicjujemy go jedynką (neutralny element mnożenia)
                Grad[0, 0] = 1.0;
            }

            // 3. PROPAGACJA WSTECZNA
            // Przechodzimy przez graf i dla każdego węzła odpalamy jego "przepis na pochodną"
            foreach (var node in topo)
            {
                node._backwardAction?.Invoke(node);
            }

            // 4. ZWROT GRAFU
            // Zwracamy listę wszystkich Tensorów, które brały udział w obliczeniach,
            // aby Janitor mógł je bezpiecznie usunąć z RAM-u po kroku optymalizatora.
            return topo;
        }

        public void Dispose()
        {
            Data?.Dispose();
            Grad?.Dispose();
            _dependencies.Clear();
            _backwardAction = null;
        }

        public static Tensor CreateOperationResult(FastMatrix<double> data, List<Tensor> dependencies, Action<Tensor> backwardAction)
        {
            return new Tensor(data, dependencies, backwardAction);
        }

    }
}