using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Data.Contracts
{
    /// <summary>
    /// Reprezentuje stan danych wewnątrz potoku przetwarzania.
    /// Implementuje IDisposable, aby zapewnić czyszczenie ciężkich buforów FastTensor.
    /// 
    /// Kontrakt własności: PipelineContext jest właścicielem swoich tensorów.
    /// Warstwa, która podmienia tensor, jest odpowiedzialna za Dispose starego.
    /// Warstwa, która tworzy nowy kontekst, NIE powinna Disposować starego kontekstu —
    /// tylko jego tensory (jeśli je podmieniła).
    /// </summary>
    public sealed class PipelineContext : IDisposable
    {
        private bool _disposed;

        /// <summary>
        /// Macierz cech wejściowych (Input).
        /// </summary>
        public FastTensor<float> Features { get; set; }

        /// <summary>
        /// Macierz wartości docelowych (Target).
        /// </summary>
        public FastTensor<float> Targets { get; set; }

        /// <summary>
        /// Metadane diagnostyczne — wypełniane przez DataPipeline po każdym kroku.
        /// </summary>
        public List<LayerDiagnostic> Diagnostics { get; } = [];

        public PipelineContext(FastTensor<float> features, FastTensor<float> targets)
        {
            Features = features ?? throw new ArgumentNullException(nameof(features));
            Targets = targets ?? throw new ArgumentNullException(nameof(targets));
        }

        /// <summary>
        /// Podmienia Features na nowy tensor i zwalnia stary.
        /// Centralizuje logikę "swap + dispose" — eliminuje rozrzucone
        /// context.Features.Dispose() + context.Features = newTensor po warstwach.
        /// </summary>
        public void ReplaceFeatures(FastTensor<float> newFeatures)
        {
            ArgumentNullException.ThrowIfNull(newFeatures);

            var old = Features;
            Features = newFeatures;

            if (!ReferenceEquals(old, newFeatures))
            {
                old?.Dispose();
            }
        }

        /// <summary>
        /// Podmienia oba tensory na nowe i zwalnia stare.
        /// Używane przez warstwy filtrujące wiersze (np. TechnicalSanityLayer, DuplicateRowFilter).
        /// </summary>
        public void ReplaceAll(FastTensor<float> newFeatures, FastTensor<float> newTargets)
        {
            ArgumentNullException.ThrowIfNull(newFeatures);
            ArgumentNullException.ThrowIfNull(newTargets);

            var oldF = Features;
            var oldT = Targets;

            Features = newFeatures;
            Targets = newTargets;

            if (!ReferenceEquals(oldF, newFeatures))
            {
                oldF?.Dispose();
            }

            if (!ReferenceEquals(oldT, newTargets))
            {
                oldT?.Dispose();
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            Features?.Dispose();
            Targets?.Dispose();

            Features = null;
            Targets = null;
        }
    }

}