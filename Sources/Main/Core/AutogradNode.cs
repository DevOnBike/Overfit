namespace DevOnBike.Overfit.Core
{
    public class AutogradNode : IDisposable
    {
        public FastMatrix<double> Data { get; }
        public FastMatrix<double> Grad { get; }
        public bool RequiresGrad { get; set; }

        public AutogradNode(FastMatrix<double> data, bool requiresGrad = true)
        {
            Data = data;
            RequiresGrad = requiresGrad;
            
            if (requiresGrad)
            {
                Grad = new FastMatrix<double>(data.Rows, data.Cols);
            }
        }

        public void Backward()
        {
            // Węzeł przekazuje sterowanie do aktywnego grafu
            ComputationGraph.Active.Backward(this);
        }

        public void Dispose()
        {
            Data?.Dispose();
            Grad?.Dispose();
        }
    }
}