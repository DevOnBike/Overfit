// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// MLP Autoencoder do detekcji anomalii metryk K8s.
    ///
    /// Architektura (domyślna, inputSize=32):
    ///   Encoder: 32 → 16 → 8 → 4  (bottleneck)
    ///   Decoder:  4 →  8 → 16 → 32
    ///   Każda warstwa oprócz ostatniej: LinearLayer → BatchNorm1D → ReLU
    ///   Ostatnia warstwa dekodera: LinearLayer bez aktywacji (rekonstrukcja znorm. wartości)
    ///
    /// Wejście:  output FeatureExtractor po RobustScaler — flat float[inputSize]
    /// Wyjście:  rekonstrukcja float[inputSize]
    ///
    /// Inference: <see cref="Reconstruct"/> — zero-alokacyjna ścieżka SIMD przez LinearLayer.
    ///   Wywołaj <see cref="Eval"/> przed użyciem produkcyjnym.
    ///
    /// Trening:
    /// <code>
    ///   autoencoder.Train();
    ///   using var graph  = new ComputationGraph();
    ///   using var input  = new AutogradNode(new FastTensor&lt;float&gt;(1, inputSize), false);
    ///   features.CopyTo(input.Data.AsSpan());
    ///   var recon = autoencoder.Forward(graph, input);
    ///   var loss  = TensorMath.MSELoss(graph, recon, input);
    ///   loss.Backward();
    ///   optimizer.Step();
    /// </code>
    /// </summary>
    public sealed class AnomalyAutoencoder : IModule
    {
        private readonly Sequential _encoder;
        private readonly Sequential _decoder;

        // Prealokowany węzeł wejściowy [1, InputSize] — reużywany na każde wywołanie Reconstruct.
        // Kształt [1, x] aktywuje zero-alokacyjną ścieżkę SIMD w LinearLayer (batchSize=1).
        private readonly AutogradNode _inputNode;

        public int InputSize { get; }
        public int BottleneckDim { get; }
        public int Hidden1 { get; }
        public int Hidden2 { get; }
        public bool IsTraining { get; private set; } = true;

        // -------------------------------------------------------------------------
        // Konstruktor
        // -------------------------------------------------------------------------

        /// <param name="inputSize">
        ///   Rozmiar wektora wejściowego = FeatureExtractor.OutputSize(featureCount).
        ///   Domyślnie 32 (8 cech × 4 statystyki).
        /// </param>
        /// <param name="hidden1">Rozmiar pierwszej ukrytej warstwy. Domyślnie inputSize/2.</param>
        /// <param name="hidden2">Rozmiar drugiej ukrytej warstwy. Domyślnie inputSize/4.</param>
        /// <param name="bottleneckDim">Wymiar bottleneck (przestrzeń latentna). Domyślnie inputSize/8.</param>
        public AnomalyAutoencoder(
            int inputSize,
            int hidden1 = 0, // 0 = inputSize / 2
            int hidden2 = 0, // 0 = inputSize / 4
            int bottleneckDim = 0) // 0 = inputSize / 8
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);

            InputSize = inputSize;
            hidden1 = hidden1 > 0 ? hidden1 : Math.Max(1, inputSize / 2);
            hidden2 = hidden2 > 0 ? hidden2 : Math.Max(1, inputSize / 4);
            bottleneckDim = bottleneckDim > 0 ? bottleneckDim : Math.Max(1, inputSize / 8);

            Hidden1 = hidden1;
            Hidden2 = hidden2;
            BottleneckDim = bottleneckDim;

            _encoder = new Sequential(
                new LinearLayer(inputSize, hidden1),
                new BatchNorm1D(hidden1),
                new ReluActivation(),

                new LinearLayer(hidden1, hidden2),
                new BatchNorm1D(hidden2),
                new ReluActivation(),

                new LinearLayer(hidden2, bottleneckDim),
                new BatchNorm1D(bottleneckDim),
                new ReluActivation()
            );

            _decoder = new Sequential(
                new LinearLayer(bottleneckDim, hidden2),
                new BatchNorm1D(hidden2),
                new ReluActivation(),

                new LinearLayer(hidden2, hidden1),
                new BatchNorm1D(hidden1),
                new ReluActivation(),

                // Ostatnia warstwa bez aktywacji — rekonstruuje znorm. wartości ∈ (-∞, +∞)
                new LinearLayer(hidden1, inputSize)
            );

            // [1, InputSize] — kształt wymagany przez SIMD inference path w LinearLayer
            _inputNode = new AutogradNode(new FastTensor<float>(1, inputSize), requiresGrad: false);
        }

        // -------------------------------------------------------------------------
        // Train / Eval
        // -------------------------------------------------------------------------

        public void Train()
        {
            IsTraining = true;
            _encoder.Train();
            _decoder.Train();
        }

        public void Eval()
        {
            IsTraining = false;
            _encoder.Eval();
            _decoder.Eval();
        }

        // -------------------------------------------------------------------------
        // Inference — zero-alokacyjna ścieżka
        // -------------------------------------------------------------------------

        /// <summary>
        /// Rekonstruuje wejście przez autoencoder.
        /// Używa prealokowanego _inputNode [1, InputSize] → SIMD path w LinearLayer.
        ///
        /// WAŻNE: wywołaj <see cref="Eval"/> przed pierwszym użyciem produkcyjnym,
        /// żeby BatchNorm używał running statistics zamiast batch statistics.
        /// </summary>
        /// <param name="features">Znormalizowane cechy z RobustScaler. Rozmiar musi być == InputSize.</param>
        /// <param name="reconstruction">Caller-owned bufor wynikowy. Rozmiar musi być >= InputSize.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Reconstruct(ReadOnlySpan<float> features, Span<float> reconstruction)
        {
            if (features.Length != InputSize)
            {
                throw new ArgumentException(
                    $"Oczekiwano {InputSize} cech, otrzymano {features.Length}.",
                    nameof(features));
            }

            if (reconstruction.Length < InputSize)
            {
                throw new ArgumentException(
                    $"Bufor rekonstrukcji za krótki: potrzeba {InputSize}, dostępne {reconstruction.Length}.",
                    nameof(reconstruction));
            }

            // Kopiuj do prealokowanego tensora — jedna kopia na wywołanie, brak alokacji
            features.CopyTo(_inputNode.Data.AsSpan());

            // graph=null → zero-alokacyjna ścieżka SIMD we wszystkich LinearLayer
            var latent = _encoder.Forward(null, _inputNode);
            var output = _decoder.Forward(null, latent);

            // Skopiuj wynik przed następnym wywołaniem Forward (output node jest reużywany przez LinearLayer)
            output.Data.AsSpan().CopyTo(reconstruction);
        }

        // -------------------------------------------------------------------------
        // Training forward — pełny autograd
        // -------------------------------------------------------------------------

        /// <summary>
        /// Forward pass z opcjonalnym nagrywaniem do grafu obliczeniowego.
        /// Używaj podczas treningu:
        ///   var recon = autoencoder.Forward(graph, inputNode);
        ///   var loss  = TensorMath.MSELoss(graph, recon, inputNode);
        /// </summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var latent = _encoder.Forward(graph, input);
            return _decoder.Forward(graph, latent);
        }

        // -------------------------------------------------------------------------
        // Parametry, serializacja, diagnostyka
        // -------------------------------------------------------------------------

        /// <summary>
        /// Zwraca wszystkie uczące się parametry encodera i dekodera.
        /// Przekaż do optymalizatora: <c>new Adam(autoencoder.Parameters(), lr)</c>.
        /// </summary>
        public IEnumerable<AutogradNode> Parameters()
        {
            foreach (var p in _encoder.Parameters()) { yield return p; }
            foreach (var p in _decoder.Parameters()) { yield return p; }
        }

        /// <summary>Liczba uczących się parametrów (wagi + biasy + gamma + beta BN).</summary>
        public int ParameterCount => Parameters().Sum(p => p.Data.Size);

        public void Save(BinaryWriter bw)
        {
            _encoder.Save(bw);
            _decoder.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            _encoder.Load(br);
            _decoder.Load(br);
        }

        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);
            Save(bw);
        }

        public void Load(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Brak pliku modelu: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);
            Load(br);
        }

        private bool _disposed;

        public void Dispose()
        {
            if (_disposed) { return; }
            _disposed = true;
            _encoder.Dispose();
            _decoder.Dispose();
            _inputNode.Dispose();
        }
    }
}