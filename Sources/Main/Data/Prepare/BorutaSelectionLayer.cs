using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Data.Prepare
{
    public sealed class BorutaSelectionLayer : IDataLayer
    {
        private readonly int _numIterations;
        private readonly int _numTrees;
        private readonly int _maxDepth;
        private readonly float _confirmationRatio;

        /// <param name="iterations">Liczba rund Boruty (im więcej, tym stabilniejsza selekcja)</param>
        /// <param name="numTrees">Liczba drzew w lesie per iteracja</param>
        /// <param name="maxDepth">Maksymalna głębokość drzew</param>
        /// <param name="confirmationRatio">Próg potwierdzenia cechy (0.5 = cecha musi wygrać w >50% iteracji)</param>
        public BorutaSelectionLayer(
            int iterations = 20,
            int numTrees = 100,
            int maxDepth = 8,
            float confirmationRatio = 0.5f)
        {
            if (iterations < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(iterations), "Liczba iteracji musi być >= 1.");
            }

            if (confirmationRatio is <= 0f or >= 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(confirmationRatio), "Próg potwierdzenia musi być w zakresie (0, 1).");
            }

            _numIterations = iterations;
            _numTrees = numTrees;
            _maxDepth = maxDepth;
            _confirmationRatio = confirmationRatio;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            if (cols == 0 || rows == 0)
            {
                return context;
            }

            // Licznik trafień per cecha — zwolniony automatycznie po pętli
            using var hitCounts = new FastBuffer<int>(cols);

            for (var iter = 0; iter < _numIterations; iter++)
            {
                // 1. Rozszerzony dataset [Oryginały | Cienie]
                using var extendedFeatures = CreateShadowDataset(context.Features);

                var forest = new FastRandomForest(numTrees: _numTrees, maxDepth: _maxDepth);

                // 2. Ważność cech z rozszerzonego datasetu
                var importance = forest.TrainAndGetImportance(extendedFeatures, context.Targets);

                // 3. Próg = maksymalna ważność wśród cieni (kolumny cols..cols*2-1)
                var shadowMax = float.MinValue;
                for (var i = cols; i < cols * 2; i++)
                {
                    if (importance[i] > shadowMax)
                    {
                        shadowMax = importance[i];
                    }
                }

                // 4. Zliczamy trafienia: cecha pobiła najlepszy cień
                for (var i = 0; i < cols; i++)
                {
                    if (importance[i] > shadowMax)
                    {
                        hitCounts[i]++;
                    }
                }
            }

            // 5. Selekcja potwierdzonych cech
            var threshold = (int)(_numIterations * _confirmationRatio);
            var keptIndices = new List<int>(cols);
            var hitSpan = hitCounts.AsReadOnlySpan();

            for (var i = 0; i < cols; i++)
            {
                if (hitSpan[i] > threshold)
                {
                    keptIndices.Add(i);
                }
            }

            // Zabezpieczenie: jeśli żadna cecha nie przeszła lub wszystkie przeszły
            if (keptIndices.Count == 0 || keptIndices.Count == cols)
            {
                return context;
            }

            // 6. Budowa nowego tensora z wybranymi kolumnami
            var filteredFeatures = ExtractSelectedColumns(context.Features, keptIndices);

            context.Features.Dispose();

            return new PipelineContext(filteredFeatures, context.Targets);
        }

        private FastTensor<float> CreateShadowDataset(FastTensor<float> original)
        {
            var rows = original.GetDim(0);
            var cols = original.GetDim(1);
            var extendedCols = cols * 2;
            var extended = new FastTensor<float>(rows, extendedCols);

            var srcSpan = original.AsSpan();
            var dstSpan = extended.AsSpan();

            // Kopiujemy oryginały do lewej połowy
            for (var r = 0; r < rows; r++)
            {
                srcSpan.Slice(r * cols, cols).CopyTo(dstSpan.Slice(r * extendedCols, cols));
            }

            // Tworzymy cienie (permutacja kolumnowa) w prawej połowie
            using var shadowBuffer = new FastBuffer<float>(rows);

            for (var c = 0; c < cols; c++)
            {
                var shadowSpan = shadowBuffer.AsSpan();

                // Kopiujemy kolumnę do bufora
                for (var r = 0; r < rows; r++)
                {
                    shadowSpan[r] = srcSpan[r * cols + c];
                }

                // Fisher-Yates shuffle
                for (var i = rows - 1; i > 0; i--)
                {
                    var j = Random.Shared.Next(i + 1);
                    (shadowSpan[i], shadowSpan[j]) = (shadowSpan[j], shadowSpan[i]);
                }

                // Wpisujemy zpermutowaną kolumnę do prawej połowy
                for (var r = 0; r < rows; r++)
                {
                    dstSpan[r * extendedCols + cols + c] = shadowSpan[r];
                }
            }

            return extended;
        }

        private FastTensor<float> ExtractSelectedColumns(FastTensor<float> src, List<int> indices)
        {
            var rows = src.GetDim(0);
            var oldCols = src.GetDim(1);
            var newCols = indices.Count;

            var result = new FastTensor<float>(rows, newCols);
            var srcSpan = src.AsSpan();
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
    }
}