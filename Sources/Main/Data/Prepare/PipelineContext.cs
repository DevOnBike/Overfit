using DevOnBike.Overfit.Core;
namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    /// Reprezentuje stan danych wewnątrz potoku przetwarzania.
    /// Implementuje IDisposable, aby zapewnić czyszczenie ciężkich buforów FastTensor.
    /// </summary>
    public class PipelineContext : IDisposable
    {
        /// <summary>
        /// Macierz cech wejściowych (Input).
        /// </summary>
        public FastTensor<float> Features { get; set; }

        /// <summary>
        /// Macierz wartości docelowych (Target).
        /// </summary>
        public FastTensor<float> Targets { get; set; }

        public PipelineContext(FastTensor<float> features, FastTensor<float> targets)
        {
            Features = features ?? throw new ArgumentNullException(nameof(features));
            Targets = targets ?? throw new ArgumentNullException(nameof(targets));
        }

        /// <summary>
        /// Zwalnia zasoby obu tensorów. 
        /// Kluczowe w potoku, gdy warstwy tworzą nowe kopie danych.
        /// </summary>
        public void Dispose()
        {
            Features?.Dispose();
            Targets?.Dispose();
        }
    }

}
