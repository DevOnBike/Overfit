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

        // Zmieniamy na listę, żeby wiedzieć co usunąć po batchu
        public List<Tensor> Backward()
        {
            var topo = new List<Tensor>();
            var visited = new HashSet<Tensor>();
            var stack = new Stack<Tensor>();

            stack.Push(this);
            visited.Add(this);

            while (stack.Count > 0)
            {
                var node = stack.Peek();
                var allChildrenProcessed = true;
                
                foreach (var child in node._dependencies)
                {
                    if (!visited.Contains(child))
                    {
                        visited.Add(child);
                        stack.Push(child);
                        allChildrenProcessed = false;
                    }
                }

                if (allChildrenProcessed)
                {
                    stack.Pop();
                    if (!topo.Contains(node)) topo.Add(node);
                }
            }

            topo.Reverse();
            // Najpierw zerujemy gradient wyjściowy (root)
            Grad.AsSpan().Fill(0);
            Grad[0, 0] = 1.0;

            foreach (var node in topo)
            {
                node._backwardAction?.Invoke(node);
            }

            return topo; // Zwracamy listę wszystkich węzłów grafu!
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