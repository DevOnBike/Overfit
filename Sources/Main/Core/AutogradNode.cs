namespace DevOnBike.Overfit.Core
{
    public class AutogradNode : IDisposable
    {
        public FastMatrix<float> Data { get; }
        public FastMatrix<float> Grad { get; }
        public bool RequiresGrad { get; set; }

        public AutogradNode(FastMatrix<float> data, bool requiresGrad = true)
        {
            Data = data;
            RequiresGrad = requiresGrad;
            
            if (requiresGrad)
            {
                Grad = new FastMatrix<float>(data.Rows, data.Cols);
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