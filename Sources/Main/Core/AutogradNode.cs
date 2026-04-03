namespace DevOnBike.Overfit.Core
{
    public sealed class AutogradNode : IDisposable
    {
        public FastTensor<float> Data { get; }
        public FastTensor<float> Grad { get; private set; }
        public bool RequiresGrad { get; set; }

        public AutogradNode(FastTensor<float> data, bool requiresGrad = true)
        {
            Data = data;
            RequiresGrad = requiresGrad;

            if (requiresGrad)
            {
                Grad = FastTensor<float>.SameShape(data, clearMemory: true);
            }
        }

        // DODANE: Pobiera wartość skalarną (np. dla węzła Loss)
        // Ponieważ FastTensor ma indexer [int i], Data[0] jest bezpieczne i szybkie.
        public float Forward() => Data[0];

        public void Backward() => ComputationGraph.Active?.Backward(this);

        public void Dispose()
        {
            Data?.Dispose();
            Grad?.Dispose();
        }
    }
}