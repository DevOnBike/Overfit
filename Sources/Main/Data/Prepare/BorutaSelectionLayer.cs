using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Data.Prepare
{
    public class BorutaSelectionLayer : IDataLayer
    {
        private readonly int _numIterations;

        public BorutaSelectionLayer(int iterations = 10) => _numIterations = iterations;

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            // 1. Tworzymy rozszerzony dataset [Oryginały | Cienie]
            using var extendedFeatures = CreateShadowDataset(context.Features);

            var forest = new FastRandomForest(numTrees: 100, maxDepth: 8);

            // 2. Pobieramy ważność cech (Ranking)
            var importance = forest.TrainAndGetImportance(extendedFeatures, context.Targets);

            // 3. Statystyka Boruty: Szukamy progu (Najlepszy z cieni)
            float shadowMax = 0;
            for (var i = cols; i < cols * 2; i++)
                if (importance[i] > shadowMax) shadowMax = importance[i];

            // 4. Selekcja zwycięzców
            var keptIndices = new List<int>();
            for (var i = 0; i < cols; i++)
            {
                // Cecha musi być wyraźnie lepsza od szumu (cienia)
                if (importance[i] > shadowMax) keptIndices.Add(i);
            }

            // 5. Budowa nowego, zoptymalizowanego tensora
            var filteredFeatures = ExtractSelectedColumns(context.Features, keptIndices);

            // Zwalniamy stare dane (Zero-GC)
            context.Features.Dispose();

            return new PipelineContext(filteredFeatures, context.Targets);
        }

        private FastTensor<float> CreateShadowDataset(FastTensor<float> original)
        {
            var rows = original.GetDim(0);
            var cols = original.GetDim(1);
            var extended = new FastTensor<float>(rows, cols * 2);

            var srcSpan = original.AsSpan();
            var dstSpan = extended.AsSpan();

            // Kopiujemy oryginały
            for (var r = 0; r < rows; r++)
                srcSpan.Slice(r * cols, cols).CopyTo(dstSpan.Slice(r * (cols * 2), cols));

            // Tworzymy cienie (Kopia + Shuffle)
            for (var c = 0; c < cols; c++)
            {
                var shadowCol = new float[rows];
                for (var r = 0; r < rows; r++) shadowCol[r] = srcSpan[r * cols + c];

                Shuffle(shadowCol); // Fisher-Yates shuffle

                for (var r = 0; r < rows; r++)
                    dstSpan[r * (cols * 2) + cols + c] = shadowCol[r];
            }

            return extended;
        }

        private void Shuffle(float[] array)
        {
            for (var i = array.Length - 1; i > 0; i--)
            {
                var j = Random.Shared.Next(i + 1);
                
                (array[i], array[j]) = (array[j], array[i]);
            }
        }

        private FastTensor<float> ExtractSelectedColumns(FastTensor<float> src, List<int> indices)
        {
            var res = new FastTensor<float>(src.GetDim(0), indices.Count);
            // Logika kopiowania kolumn...
            return res;
        }
    }
}
