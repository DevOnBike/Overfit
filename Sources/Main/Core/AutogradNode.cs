namespace DevOnBike.Overfit.Core
{
    public class AutogradNode : IDisposable
    {
        // Migracja na FastTensor
        public FastTensor<float> Data { get; }
        public FastTensor<float>? Grad { get; private set; }
        public bool RequiresGrad { get; set; }

        public AutogradNode(FastTensor<float> data, bool requiresGrad = true)
        {
            Data = data;
            RequiresGrad = requiresGrad;

            if (requiresGrad)
            {
                // Inicjalizacja gradientu o identycznym kształcie co dane
                Grad = new FastTensor<float>(data.Shape);

                // Bezwzględne czyszczenie - pamięć z ArrayPool może zawierać śmieci!
                Grad.AsSpan().Clear();
            }
        }

        public void Backward()
        {
            ComputationGraph.Active?.Backward(this);
        }

        public void Dispose()
        {
            Data?.Dispose();
            Grad?.Dispose();
        }
    }
}