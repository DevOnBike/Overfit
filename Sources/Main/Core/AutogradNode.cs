namespace DevOnBike.Overfit.Core
{
    public class AutogradNode : IDisposable
    {
        public FloatFastMatrix Data { get; }
        public FloatFastMatrix Grad { get; }
        public bool RequiresGrad { get; set; }

        public AutogradNode(FloatFastMatrix data, bool requiresGrad = true)
        {
            Data = data;
            RequiresGrad = requiresGrad;
            
            if (requiresGrad)
            {
                Grad = new FloatFastMatrix(data.Rows, data.Cols);
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