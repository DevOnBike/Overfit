using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    /// Wykrywa i usuwa zduplikowane wiersze na podstawie Features.
    /// Używa dwuetapowego porównania: szybki hash → pełne porównanie Span,
    /// aby uniknąć fałszywych kolizji.
    /// </summary>
    public sealed class DuplicateRowFilterLayer : IDataLayer
    {
        private readonly bool _includeTargetInComparison;

        /// <param name="includeTargetInComparison">
        /// Czy uwzględniać kolumnę Target przy porównaniu duplikatów.
        /// true = dwa wiersze z identycznymi Features ale różnym Target NIE są duplikatami.
        /// false = porównujemy tylko Features (domyślne — bezpieczniejsze dla regresji).
        /// </param>
        public DuplicateRowFilterLayer(bool includeTargetInComparison = false)
        {
            _includeTargetInComparison = includeTargetInComparison;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            if (rows <= 1)
            {
                return context;
            }

            var featureSpan = context.Features.AsReadOnlySpan();
            var targetSpan = context.Targets.AsReadOnlySpan();

            // Etap 1: Obliczamy hash per wiersz
            using var rowHashes = new FastBuffer<int>(rows);
            var hashSpan = rowHashes.AsSpan();

            for (var r = 0; r < rows; r++)
            {
                hashSpan[r] = ComputeRowHash(featureSpan, targetSpan, r, cols);
            }

            // Etap 2: Grupujemy po hashu, potem weryfikujemy pełnym porównaniem
            // Zachowujemy pierwsze wystąpienie, wyrzucamy kolejne
            var keptIndices = new List<int>(rows);
            var hashBuckets = new Dictionary<int, List<int>>(rows);

            for (var r = 0; r < rows; r++)
            {
                var hash = hashSpan[r];

                if (!hashBuckets.TryGetValue(hash, out var bucket))
                {
                    // Nowy hash — na pewno unikat
                    bucket = new List<int>(1);
                    hashBuckets[hash] = bucket;
                    bucket.Add(r);
                    keptIndices.Add(r);
                    continue;
                }

                // Kolizja hashowa — porównujemy z każdym istniejącym wierszem w kubełku
                var isDuplicate = false;
                foreach (var existingRow in bucket)
                {
                    if (RowsEqual(featureSpan, targetSpan, existingRow, r, cols))
                    {
                        isDuplicate = true;
                        break;
                    }
                }

                if (!isDuplicate)
                {
                    bucket.Add(r);
                    keptIndices.Add(r);
                }
            }

            // Brak duplikatów — zwracamy bez alokacji nowych tensorów
            if (keptIndices.Count == rows)
            {
                return context;
            }

            // Budujemy nowe tensory z unikatowych wierszy
            var newRows = keptIndices.Count;
            var newFeatures = new FastTensor<float>(newRows, cols);
            var newTargets = new FastTensor<float>(newRows, 1);

            var dstFeatures = newFeatures.AsSpan();
            var dstTargets = newTargets.AsSpan();

            for (var i = 0; i < newRows; i++)
            {
                var srcRow = keptIndices[i];

                featureSpan.Slice(srcRow * cols, cols).CopyTo(dstFeatures.Slice(i * cols, cols));
                dstTargets[i] = targetSpan[srcRow];
            }

            context.Features.Dispose();
            context.Targets.Dispose();

            return new PipelineContext(newFeatures, newTargets);
        }

        private int ComputeRowHash(
            ReadOnlySpan<float> features,
            ReadOnlySpan<float> targets,
            int row,
            int cols)
        {
            var hash = new HashCode();
            var offset = row * cols;

            // Hashujemy Features blokach po 4 dla lepszego pipeline'owania CPU
            var c = 0;
            var limit = cols - 3;
            for (; c < limit; c += 4)
            {
                hash.Add(features[offset + c]);
                hash.Add(features[offset + c + 1]);
                hash.Add(features[offset + c + 2]);
                hash.Add(features[offset + c + 3]);
            }

            // Reszta
            for (; c < cols; c++)
            {
                hash.Add(features[offset + c]);
            }

            if (_includeTargetInComparison)
            {
                hash.Add(targets[row]);
            }

            return hash.ToHashCode();
        }

        private bool RowsEqual(
            ReadOnlySpan<float> features,
            ReadOnlySpan<float> targets,
            int rowA,
            int rowB,
            int cols)
        {
            var offsetA = rowA * cols;
            var offsetB = rowB * cols;

            // Porównanie bitowe przez SequenceEqual — szybsze niż element-po-elemencie
            // i poprawnie traktuje NaN (NaN == NaN na poziomie bitów w SequenceEqual)
            if (!features.Slice(offsetA, cols).SequenceEqual(features.Slice(offsetB, cols)))
            {
                return false;
            }

            if (_includeTargetInComparison)
            {
                return targets[rowA] == targets[rowB];
            }

            return true;
        }
    }
}