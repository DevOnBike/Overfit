using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    /// Pierwsza warstwa potoku — sanityzacja technicznych artefaktów w danych.
    /// Czyści NaN, ±Infinity, subnormale i opcjonalnie wyrzuca wiersze z nadmiernym "zanieczyszczeniem".
    /// Operuje in-place na Spanach (zero alokacji dla czyszczenia).
    /// </summary>
    public sealed class TechnicalSanityLayer : IDataLayer
    {
        private readonly float _maxCorruptedRatio;
        private readonly float _replacementValue;

        /// <param name="maxCorruptedRatio">
        /// Maksymalny dopuszczalny udział uszkodzonych wartości w wierszu (0.0–1.0).
        /// Wiersze przekraczające próg zostaną odrzucone.
        /// Wartość 1.0 = nigdy nie wyrzucaj wierszy (tylko czyść in-place).
        /// </param>
        /// <param name="replacementValue">Wartość wstawiana w miejsce NaN/Inf (domyślnie 0)</param>
        public TechnicalSanityLayer(float maxCorruptedRatio = 1.0f, float replacementValue = 0f)
        {
            if (maxCorruptedRatio is < 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(maxCorruptedRatio),
                    "Próg uszkodzonych wartości musi być w zakresie [0, 1].");
            }

            _maxCorruptedRatio = maxCorruptedRatio;
            _replacementValue = replacementValue;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            if (rows == 0 || cols == 0)
            {
                return context;
            }

            // Targets: zawsze czyścimy in-place (1 kolumna, nie ma sensu wyrzucać wiersza na tej podstawie)
            CleanSpan(context.Targets.AsSpan());

            // Tryb prosty: czyść in-place bez filtrowania wierszy
            if (_maxCorruptedRatio >= 1.0f)
            {
                CleanSpan(context.Features.AsSpan());
                return context;
            }

            // Tryb z filtrowaniem: identyfikuj czyste wiersze, wyrzuć brudne
            var featureSpan = context.Features.AsSpan();
            var maxCorruptedPerRow = (int)(cols * _maxCorruptedRatio);

            // Zbieramy indeksy wierszy, które przechodzą próg
            using var corruptedCounts = new FastBuffer<int>(rows);
            var countSpan = corruptedCounts.AsSpan();

            for (var r = 0; r < rows; r++)
            {
                var offset = r * cols;
                var corrupted = 0;

                for (var c = 0; c < cols; c++)
                {
                    if (IsCorrupted(featureSpan[offset + c]))
                    {
                        corrupted++;
                    }
                }

                countSpan[r] = corrupted;
            }

            // Zbieramy indeksy wierszy do zachowania
            var keptIndices = new List<int>(rows);
            for (var r = 0; r < rows; r++)
            {
                if (countSpan[r] <= maxCorruptedPerRow)
                {
                    keptIndices.Add(r);
                }
            }

            // Wszystkie wiersze przeszły — czyścimy in-place i zwracamy
            if (keptIndices.Count == rows)
            {
                CleanSpan(featureSpan);
                return context;
            }

            // Zero wierszy przeszło — skrajny przypadek, zachowujemy wszystkie i czyścimy
            if (keptIndices.Count == 0)
            {
                CleanSpan(featureSpan);
                return context;
            }

            // Budujemy nowe tensory tylko z czystych wierszy
            var newRows = keptIndices.Count;
            var newFeatures = new FastTensor<float>(newRows, cols);
            var newTargets = new FastTensor<float>(newRows, 1);

            var srcFeatures = context.Features.AsSpan();
            var srcTargets = context.Targets.AsSpan();
            var dstFeatures = newFeatures.AsSpan();
            var dstTargets = newTargets.AsSpan();

            for (var i = 0; i < newRows; i++)
            {
                var srcRow = keptIndices[i];

                // Kopiujemy wiersz Features
                srcFeatures.Slice(srcRow * cols, cols).CopyTo(dstFeatures.Slice(i * cols, cols));

                // Kopiujemy wiersz Targets
                dstTargets[i] = srcTargets[srcRow];
            }

            // Czyścimy resztki NaN/Inf w zachowanych wierszach (mogą mieć < maxCorrupted)
            CleanSpan(dstFeatures);
            CleanSpan(dstTargets);

            // Zwalniamy stare tensory
            context.Features.Dispose();
            context.Targets.Dispose();

            return new PipelineContext(newFeatures, newTargets);
        }

        private void CleanSpan(Span<float> span)
        {
            for (var i = 0; i < span.Length; i++)
            {
                if (IsCorrupted(span[i]))
                {
                    span[i] = _replacementValue;
                }
            }
        }

        /// <summary>
        /// Sprawdza czy wartość jest technicznie uszkodzona:
        /// NaN, ±Infinity lub subnormalna (denormalized).
        /// Subnormale mogą powodować 100x spowolnienie operacji FP na niektórych CPU.
        /// </summary>
        private static bool IsCorrupted(float value)
        {
            return float.IsNaN(value)
                || float.IsInfinity(value)
                || float.IsSubnormal(value);
        }
    }
}