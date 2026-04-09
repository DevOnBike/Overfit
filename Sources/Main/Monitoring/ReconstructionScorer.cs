// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Zamienia błąd rekonstrukcji autoenkodera na interpretowalny <c>AnomalyScore ∈ [0, 1]</c>.
    ///
    /// Zasada działania:
    ///   Autoencoder trenowany na normalnym ruchu potrafi go dobrze rekonstruować (niski MSE).
    ///   Anomalia powoduje wzrost MSE — scorer normalizuje go względem progu wyznaczonego
    ///   z danych normalnych:
    ///
    ///   AnomalyScore = Clamp(MSE(input, reconstruction) / Threshold, 0, 1)
    ///
    ///   - 0.0 → wzorzec w pełni znany modelowi, normalne zachowanie
    ///   - 0.5 → połowa progu kalibracyjnego
    ///   - 1.0 → MSE osiągnął lub przekroczył p99 z fazy kalibracji → anomalia
    ///
    /// Użycie w pipeline:
    /// <code>
    ///   // Inicjalizacja (raz):
    ///   var scorer = new ReconstructionScorer();
    ///   scorer.CalibrateFromModel(autoencoder, normalFeatureVectors);
    ///
    ///   // Online scoring (każde okno):
    ///   autoencoder.Reconstruct(features, reconstructionScratch);
    ///   var score = scorer.Score(features, reconstructionScratch);
    ///   if (score >= alertThreshold) alertEngine.Trigger(score);
    /// </code>
    /// </summary>
    public sealed class ReconstructionScorer
    {
        // Przy rozmiarze ≤ 512 floatów (2 KB) używamy stackalloc dla bufora diff.
        // inputSize=32 (8 cech × 4 statystyki) = 128 B — zdecydowanie w zakresie.
        private const int StackAllocThreshold = 512;

        /// <summary>
        /// Próg MSE odpowiadający p99 z danych kalibracyjnych (normalny ruch).
        /// Domyślnie 1.0 — przed kalibracją score = raw MSE (nienormalizowany).
        /// </summary>
        public float Threshold { get; private set; } = 1.0f;

        /// <summary>True gdy <see cref="Calibrate"/> lub <see cref="CalibrateFromModel"/> zostało wywołane.</summary>
        public bool IsCalibrated { get; private set; }

        // -------------------------------------------------------------------------
        // Score — główna ścieżka online
        // -------------------------------------------------------------------------

        /// <summary>
        /// Oblicza AnomalyScore z wektora cech i jego rekonstrukcji przez autoencoder.
        ///
        /// Zero alokacji gdy features.Length ≤ 512.
        /// </summary>
        /// <param name="features">Wejście autoenkodera (znormalizowane cechy z RobustScaler).</param>
        /// <param name="reconstruction">Wyjście autoenkodera z <see cref="AnomalyAutoencoder.Reconstruct"/>.</param>
        /// <returns>AnomalyScore ∈ [0, 1].</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Score(ReadOnlySpan<float> features, ReadOnlySpan<float> reconstruction)
        {
            var mse = ComputeMse(features, reconstruction);
            return ComputeScore(mse);
        }

        /// <summary>
        /// Normalizuje surowe MSE do [0, 1] względem skalibrowanego progu.
        /// Użyteczne gdy MSE jest już obliczone osobno (np. do logowania).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float ComputeScore(float mse)
        {
            return Math.Clamp(mse / Threshold, 0f, 1f);
        }

        // -------------------------------------------------------------------------
        // ComputeMse — statyczna, używana też przez OfflineTrainingJob
        // -------------------------------------------------------------------------

        /// <summary>
        /// Oblicza Mean Squared Error między dwoma wektorami.
        ///   MSE = (1/n) × Σ(a_i - b_i)²
        ///
        /// Zero alokacji gdy a.Length ≤ 512 (stackalloc dla bufora diff).
        /// SIMD-accelerated przez TensorPrimitives.
        /// </summary>
        /// <exception cref="ArgumentException">Gdy długości spanów się różnią lub są zerowe.</exception>
        public static float ComputeMse(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            if (a.Length != b.Length)
            {
                throw new ArgumentException(
                    $"Długości wektorów muszą być równe: a.Length={a.Length}, b.Length={b.Length}.");
            }

            if (a.Length == 0)
            {
                throw new ArgumentException("Wektory nie mogą być puste.");
            }

            if (a.Length <= StackAllocThreshold)
            {
                Span<float> diff = stackalloc float[a.Length];
                return ComputeMseCore(a, b, diff);
            }

            using var rented = new FastBuffer<float>(a.Length);
            return ComputeMseCore(a, b, rented.AsSpan());
        }

        // -------------------------------------------------------------------------
        // Calibrate — kalibracja progu z gotowych wartości MSE
        // -------------------------------------------------------------------------

        /// <summary>
        /// Kalibruje próg anomalii jako percentyl p wartości MSE z normalnych danych.
        ///
        /// Typowe użycie: zbierz MSE dla N normalnych okien podczas offline training,
        /// wywołaj Calibrate(mseValues, percentile: 0.99f).
        ///
        /// Alternatywa: <see cref="CalibrateFromModel"/> — automatyzuje zbieranie MSE.
        /// </summary>
        /// <param name="mseValues">Wartości MSE z normalnych okien (min. 1 element).</param>
        /// <param name="percentile">Percentyl progu. Domyślnie 0.99 (p99).</param>
        /// <exception cref="ArgumentException">Gdy mseValues jest pusta lub percentyl poza (0, 1].</exception>
        public void Calibrate(ReadOnlySpan<float> mseValues, float percentile = 0.99f)
        {
            if (mseValues.IsEmpty)
            {
                throw new ArgumentException("mseValues nie może być pusty.", nameof(mseValues));
            }

            if (percentile is <= 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(percentile),
                    $"Percentyl musi być w zakresie (0, 1], otrzymano {percentile}.");
            }

            // Kopiujemy i sortujemy — Calibrate to operacja offline, alokacja dopuszczalna
            var sorted = mseValues.ToArray();
            Array.Sort(sorted);

            var index = (int)MathF.Ceiling(percentile * sorted.Length) - 1;
            index = Math.Clamp(index, 0, sorted.Length - 1);

            // Guard: próg nigdy nie może być zero — uniknięcie dzielenia przez zero w Score
            Threshold = MathF.Max(sorted[index], float.Epsilon);
            IsCalibrated = true;
        }

        // -------------------------------------------------------------------------
        // CalibrateFromModel — wygodny overload dla OfflineTrainingJob
        // -------------------------------------------------------------------------

        /// <summary>
        /// Automatycznie kalibruje próg przez uruchomienie autoenkodera na zbiorze normalnych wektorów.
        ///
        /// Dla każdego wektora: rekonstrukcja → MSE → zebranie → Calibrate(p99).
        /// Nie wymaga osobnego bufora MSE — zarządza nim wewnętrznie.
        /// </summary>
        /// <param name="autoencoder">Wytrenowany autoencoder w trybie Eval.</param>
        /// <param name="normalFeatures">
        ///   Wektory znormalizowanych cech z normalnego ruchu.
        ///   Każdy musi mieć długość <see cref="AnomalyAutoencoder.InputSize"/>.
        /// </param>
        /// <param name="percentile">Percentyl progu. Domyślnie 0.99.</param>
        /// <exception cref="InvalidOperationException">Gdy normalFeatures jest pusta.</exception>
        public void CalibrateFromModel(
            AnomalyAutoencoder autoencoder,
            IEnumerable<float[]> normalFeatures,
            float percentile = 0.99f)
        {
            ArgumentNullException.ThrowIfNull(autoencoder);
            ArgumentNullException.ThrowIfNull(normalFeatures);

            var reconstruction = new float[autoencoder.InputSize];
            var mseValues = new List<float>();

            foreach (var features in normalFeatures)
            {
                if (features.Length != autoencoder.InputSize)
                {
                    throw new ArgumentException(
                        $"Wektor cech ma długość {features.Length}, oczekiwano {autoencoder.InputSize}.",
                        nameof(normalFeatures));
                }

                autoencoder.Reconstruct(features, reconstruction);
                mseValues.Add(ComputeMse(features, reconstruction));
            }

            if (mseValues.Count == 0)
            {
                throw new InvalidOperationException("normalFeatures jest puste — brak danych do kalibracji.");
            }

            Calibrate(mseValues.ToArray().AsSpan(), percentile);
        }

        // -------------------------------------------------------------------------
        // Serializacja
        // -------------------------------------------------------------------------

        /// <summary>Zapisuje próg i flagę kalibracji do strumienia.</summary>
        public void Save(BinaryWriter bw)
        {
            bw.Write(Threshold);
            bw.Write(IsCalibrated);
        }

        /// <summary>Wczytuje próg i flagę kalibracji ze strumienia.</summary>
        public void Load(BinaryReader br)
        {
            Threshold = br.ReadSingle();
            IsCalibrated = br.ReadBoolean();
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
                throw new FileNotFoundException($"Brak pliku scorera: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);
            Load(br);
        }

        // -------------------------------------------------------------------------
        // Prywatne
        // -------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ComputeMseCore(
            ReadOnlySpan<float> a,
            ReadOnlySpan<float> b,
            Span<float> diff)
        {
            // diff = a - b  (SIMD)
            TensorPrimitives.Subtract(a, b, diff);

            // Σ(diff²) = Dot(diff, diff)  (SIMD)
            var sumSq = TensorPrimitives.Dot(diff, diff);

            return sumSq / a.Length;
        }
    }
}