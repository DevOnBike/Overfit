using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Evolutionary.Contracts
{
    public sealed class EvolutionWorkspace : IDisposable
    {
        private bool _disposed;

        public FastTensor<float> Population { get; }
        public FastTensor<float> Fitness { get; }
        public FastTensor<float> Perturbations { get; }
        public int[] Ranking { get; }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(GetType().FullName);
            }
        }
    }
}

