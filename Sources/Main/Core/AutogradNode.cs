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
                // SameShape zamiast data.Shape — eliminuje new int[Rank] per instancja.
                // clearMemory:true (domyślne) — konstruktor FastTensor już czyści ArrayPool.
                // Redundantne Grad.AsSpan().Clear() usunięte.
                Grad = FastTensor<float>.SameShape(data, clearMemory: true);
            }
        }

        public void Backward() => ComputationGraph.Active?.Backward(this);

        public void Dispose()
        {
            Data?.Dispose();
            Grad?.Dispose();
        }
    }
}