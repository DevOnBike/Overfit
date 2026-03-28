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
        
        // ZMIANA ARCHITEKTONICZNA: Delegat przyjmuje węzeł, na którym operuje
        private Action<Tensor> _backwardAction;

        public Tensor(FastMatrix<double> data, bool requiresGrad = true)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            RequiresGrad = requiresGrad;
            Grad = new FastMatrix<double>(data.Rows, data.Cols);
        }

        private Tensor(FastMatrix<double> data, List<Tensor> dependencies, Action<Tensor> backwardAction)
        {
            Data = data;
            RequiresGrad = true;
            Grad = new FastMatrix<double>(data.Rows, data.Cols);
            _dependencies = dependencies;
            _backwardAction = backwardAction;
        }

        // ====================================================================
        // MAGIA AUTOGRADA - Wsteczna Propagacja
        // ====================================================================

        public void Backward()
        {
            // UWAGA: Nie zerujemy już gradientu korzenia! 
            // Założenie architektoniczne: Funkcja Straty (lub kod testu) 
            // wstrzykuje początkowy sygnał do this.Grad PRZED wywołaniem Backward().

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

            // Odwracamy listę i odpalamy propagację wstecz!
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
            Data?.Dispose();
            Grad?.Dispose();
            _dependencies.Clear();
            _backwardAction = null;
        }
    }
}