namespace DevOnBike.Overfit.Core
{
    /// <summary>
    /// Węzeł grafu obliczeniowego. Owija FastMatrix, przechowuje gradienty 
    /// oraz historię operacji do wstecznej propagacji (Autograd).
    /// </summary>
    public sealed class AutogradNode : IDisposable
    {
        public FastMatrix<double> Data { get; }
        public FastMatrix<double> Grad { get; }
        public bool RequiresGrad { get; }

        internal readonly List<AutogradNode> _dependencies = [];
        internal Action<AutogradNode> _backwardAction;

        private readonly bool _ownsData;

        public AutogradNode(FastMatrix<double> data, bool requiresGrad = true, bool ownsData = true)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            RequiresGrad = requiresGrad;
            Grad = new FastMatrix<double>(data.Rows, data.Cols);
            _ownsData = ownsData;
        }

        private AutogradNode(FastMatrix<double> data, List<AutogradNode> dependencies, Action<AutogradNode> backwardAction)
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

        public List<AutogradNode> Backward()
        {
            // 1. BUDOWA GRAFU OBLICZENIOWEGO (Sortowanie Topologiczne)
            // Musimy przetwarzać węzły od końca (Loss) do początku (Inputs)
            var topo = new List<AutogradNode>();
            var visited = new HashSet<AutogradNode>();

            void BuildTopo(AutogradNode node)
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
                // Jeśli startujemy od macierzy, traktujemy to jak pochodną sumy wszystkich elementów
                // Wypełniamy całą przestrzeń jedynkami, a nie tylko [0,0]
                Grad.AsSpan().Fill(1.0);
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
            // Zabezpieczenie: niszczymy macierz TYLKO, jeśli węzeł jest jej wyłącznym właścicielem
            if (_ownsData)
            {
                Data?.Dispose();
            }

            // Gradient i tak jest zawsze własnością węzła, więc możemy go zwolnić
            Grad?.Dispose();

            _dependencies.Clear();
            _backwardAction = null;
        }

        public static AutogradNode CreateOperationResult(FastMatrix<double> data, List<AutogradNode> dependencies, Action<AutogradNode> backwardAction)
        {
            return new AutogradNode(data, dependencies, backwardAction);
        }

    }
}