// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    /// Filtruje kolumny o silnej korelacji liniowej (Pearson).
    /// Gdy dwie cechy są skorelowane powyżej progu, warstwa zachowuje tę
    /// z wyższą korelacją z Target (jeśli dostępna) lub pierwszą napotkaną.
    ///
    /// Motywacja:
    /// - Dwie cechy skorelowane na 0.98 nie wnoszą nowej informacji, ale podwajają wymiarowość
    /// - Multikolinearność destabilizuje gradienty — model "oscyluje" między cechami
    /// - Mniej kolumn = szybszy trening i mniej pamięci
    ///
    /// W danych nieruchomościowych typowe pary:
    /// - Powierzchnia użytkowa vs liczba pokoi (r ≈ 0.92)
    /// - Cena za m² vs cena całkowita (r ≈ 0.85 po normalizacji)
    /// - Rok budowy vs stan techniczny (r ≈ 0.78)
    ///
    /// Warstwa powinna być stosowana PO skalowaniu, a PRZED Borutą.
    /// Pearson jest wrażliwy na skalę, więc nieskalowane dane dadzą zafałszowane wyniki.
    /// 
    /// Implementuje Fit/Transform — statystyki z treningu są reużywane na inference.
    /// </summary>
    public sealed class CorrelationFilterLayer : IDataLayer
    {
        private readonly float _threshold;
        private readonly DropStrategy _strategy;

        // Zapamiętane z Fit — indeksy kolumn do zachowania
        private int[] _keptIndices;
        private bool _fitted;

        /// <param name="threshold">
        /// Próg korelacji (wartość bezwzględna |r|). Domyślnie 0.95.
        /// 0.98 = agresywna filtracja (tylko niemal identyczne cechy).
        /// 0.85 = łagodna filtracja (szersze czyszczenie multikolinearności).
        /// </param>
        /// <param name="strategy">
        /// Strategia wyboru cechy do usunięcia z pary skorelowanych.
        /// KeepFirst: zachowuje cechę z niższym indeksem (szybkie, deterministyczne).
        /// KeepHigherTargetCorrelation: zachowuje cechę silniej skorelowaną z Target
        /// (wolniejsze — wymaga N dodatkowych obliczeń Pearsona, ale daje lepszą selekcję).
        /// </param>
        public CorrelationFilterLayer(
            float threshold = 0.95f,
            DropStrategy strategy = DropStrategy.KeepHigherTargetCorrelation)
        {
            if (threshold is <= 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(threshold), "Próg korelacji musi być w zakresie (0, 1].");
            }

            _threshold = threshold;
            _strategy = strategy;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            if (rows < 3 || cols <= 1)
            {
                return context;
            }

            // Fit: identyfikujemy kolumny do zachowania
            if (!_fitted)
            {
                _keptIndices = Fit(context.Features, context.Targets, rows, cols);
                _fitted = true;
            }

            // Wszystkie kolumny przeszły — zwracamy bez zmian
            if (_keptIndices.Length == cols)
            {
                return context;
            }

            // Transform: budujemy węższy tensor
            var filtered = ExtractColumns(context.Features, _keptIndices, rows);

            context.Features.Dispose();
            context.Features = filtered;

            return context;
        }

        private int[] Fit(FastTensor<float> features, FastTensor<float> targets, int rows, int cols)
        {
            var featureSpan = features.AsReadOnlySpan();
            var targetSpan = targets.AsReadOnlySpan();

            // Prekomputujemy statystyki per kolumna (sum, sumSq, mean)
            // aby uniknąć wielokrotnego przeliczania w O(n²) parach
            using var sums = new FastBuffer<double>(cols);
            using var sumsSq = new FastBuffer<double>(cols);
            using var means = new FastBuffer<double>(cols);

            var sumSpan = sums.AsSpan();
            var sumSqSpan = sumsSq.AsSpan();
            var meanSpan = means.AsSpan();

            for (var c = 0; c < cols; c++)
            {
                double sum = 0;
                double sumSq = 0;

                for (var r = 0; r < rows; r++)
                {
                    double val = featureSpan[r * cols + c];
                    sum += val;
                    sumSq += val * val;
                }

                sumSpan[c] = sum;
                sumSqSpan[c] = sumSq;
                meanSpan[c] = sum / rows;
            }

            // Korelacje z Target (jeśli strategia tego wymaga)
            float[] targetCorrelations = null;
            if (_strategy == DropStrategy.KeepHigherTargetCorrelation)
            {
                targetCorrelations = ComputeTargetCorrelations(
                    featureSpan, targetSpan, sumSpan, sumSqSpan, rows, cols);
            }

            // Macierz korelacji — obliczamy tylko trójkąt górny
            var dropped = new HashSet<int>();

            for (var i = 0; i < cols; i++)
            {
                if (dropped.Contains(i))
                {
                    continue;
                }

                for (var j = i + 1; j < cols; j++)
                {
                    if (dropped.Contains(j))
                    {
                        continue;
                    }

                    var r = CalculatePearsonFast(
                        featureSpan, i, j, rows, cols,
                        sumSpan, sumSqSpan);

                    if (MathF.Abs(r) < _threshold)
                    {
                        continue;
                    }

                    // Para (i, j) jest skorelowana — wybieramy którą wyrzucić
                    var dropIdx = ChooseColumnToDrop(i, j, targetCorrelations);
                    dropped.Add(dropIdx);
                }
            }

            // Budujemy posortowaną listę zachowanych indeksów
            var kept = new List<int>(cols - dropped.Count);
            for (var c = 0; c < cols; c++)
            {
                if (!dropped.Contains(c))
                {
                    kept.Add(c);
                }
            }

            return kept.ToArray();
        }

        /// <summary>
        /// Pearson z prekomputowanymi sumami — unika podwójnego przeliczania.
        /// Jedyny koszt per para to obliczenie sumAB (iloczyn skalarny dwóch kolumn).
        /// Złożoność: O(rows) per para zamiast O(3 * rows) w naiwnej implementacji.
        /// </summary>
        private float CalculatePearsonFast(
            ReadOnlySpan<float> span, int colA, int colB,
            int rows, int cols,
            Span<double> sums, Span<double> sumsSq)
        {
            double sumAB = 0;

            for (var r = 0; r < rows; r++)
            {
                sumAB += (double)span[r * cols + colA] * span[r * cols + colB];
            }

            var num = (rows * sumAB) - (sums[colA] * sums[colB]);
            var denA = (rows * sumsSq[colA]) - (sums[colA] * sums[colA]);
            var denB = (rows * sumsSq[colB]) - (sums[colB] * sums[colB]);
            var den = Math.Sqrt(denA * denB);

            if (den == 0)
            {
                return 0f;
            }

            return (float)(num / den);
        }

        /// <summary>
        /// Oblicza korelację każdej cechy z Target.
        /// Cecha silniej skorelowana z Target jest "cenniejsza" — zachowujemy ją.
        /// </summary>
        private float[] ComputeTargetCorrelations(
            ReadOnlySpan<float> featureSpan,
            ReadOnlySpan<float> targetSpan,
            Span<double> featureSums,
            Span<double> featureSumsSq,
            int rows, int cols)
        {
            var result = new float[cols];

            // Statystyki Target
            double tSum = 0;
            double tSumSq = 0;

            for (var r = 0; r < rows; r++)
            {
                double val = targetSpan[r];
                tSum += val;
                tSumSq += val * val;
            }

            for (var c = 0; c < cols; c++)
            {
                double sumFT = 0;

                for (var r = 0; r < rows; r++)
                {
                    sumFT += (double)featureSpan[r * cols + c] * targetSpan[r];
                }

                var num = (rows * sumFT) - (featureSums[c] * tSum);
                var denF = (rows * featureSumsSq[c]) - (featureSums[c] * featureSums[c]);
                var denT = (rows * tSumSq) - (tSum * tSum);
                var den = Math.Sqrt(denF * denT);

                result[c] = den == 0 ? 0f : (float)(num / den);
            }

            return result;
        }

        /// <summary>
        /// Wybiera kolumnę do usunięcia z pary skorelowanych.
        /// </summary>
        private int ChooseColumnToDrop(int colA, int colB, float[] targetCorrelations)
        {
            if (_strategy == DropStrategy.KeepFirst)
            {
                // Deterministyczne — zawsze wyrzucamy późniejszą kolumnę
                return colB;
            }

            // KeepHigherTargetCorrelation: zachowujemy cechę cenniejszą dla predykcji
            var corrA = MathF.Abs(targetCorrelations[colA]);
            var corrB = MathF.Abs(targetCorrelations[colB]);

            // Przy remisie zachowujemy wcześniejszą (stabilność)
            return corrA >= corrB ? colB : colA;
        }

        private FastTensor<float> ExtractColumns(FastTensor<float> src, int[] indices, int rows)
        {
            var oldCols = src.GetDim(1);
            var newCols = indices.Length;

            var result = new FastTensor<float>(rows, newCols);
            var srcSpan = src.AsReadOnlySpan();
            var dstSpan = result.AsSpan();

            for (var r = 0; r < rows; r++)
            {
                var srcOffset = r * oldCols;
                var dstOffset = r * newCols;

                for (var c = 0; c < newCols; c++)
                {
                    dstSpan[dstOffset + c] = srcSpan[srcOffset + indices[c]];
                }
            }

            return result;
        }

        /// <summary>
        /// Resetuje zapamiętane indeksy kolumn.
        /// Wymusza ponowny Fit przy następnym wywołaniu Process.
        /// </summary>
        public void Reset()
        {
            _keptIndices = null;
            _fitted = false;
        }
    }

    public enum DropStrategy
    {
        /// <summary>Zachowuje kolumnę z niższym indeksem — szybkie, deterministyczne.</summary>
        KeepFirst,

        /// <summary>
        /// Zachowuje kolumnę silniej skorelowaną z Target — wolniejsze,
        /// ale daje lepszą selekcję pod kątem predykcji.
        /// </summary>
        KeepHigherTargetCorrelation
    }
}