using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Prepare;

namespace DevOnBike.Overfit.Data.Layers
{
    public sealed class CorrelationFilterLayer : IDataLayer
    {
        private readonly float _threshold;
        private int[] _indicesToDrop;

        public CorrelationFilterLayer(float threshold = 0.98f) => _threshold = threshold;

        public PipelineContext Process(PipelineContext context)
        {
            // Lazy Fit: Jeśli nie mamy jeszcze indeksów, wyliczamy je na aktualnych danych
            if (_indicesToDrop == null)
            {
                _indicesToDrop = IdentifyRedundantColumns(context.Features);
            }

            // Jeśli nie ma nic do wyrzucenia, zwracamy kontekst bez zmian
            if (_indicesToDrop.Length == 0) return context;

            // Transform: Tworzymy nowy, węższy tensor
            var filteredFeatures = FilterTensor(context.Features, _indicesToDrop);

            // Zwalniamy stary tensor (ważne dla ArrayPool!)
            context.Features.Dispose();

            // Podpinamy przefiltrowane dane pod kontekst
            context.Features = filteredFeatures;

            return context;
        }

        private int[] IdentifyRedundantColumns(FastTensor<float> x)
        {
            int cols = x.GetDim(1);
            var dropped = new HashSet<int>();

            for (int i = 0; i < cols; i++)
            {
                if (dropped.Contains(i)) continue;

                for (int j = i + 1; j < cols; j++)
                {
                    if (dropped.Contains(j)) continue;

                    // Obliczamy Pearsona na spanach
                    float r = CalculatePearson(x, i, j);
                    if (MathF.Abs(r) >= _threshold)
                    {
                        dropped.Add(j);
                    }
                }
            }
            // Sortujemy malejąco, żeby bezpiecznie operować na indeksach jeśli trzeba
            return dropped.OrderByDescending(idx => idx).ToArray();
        }

        private FastTensor<float> FilterTensor(FastTensor<float> x, int[] toDrop)
        {
            int rows = x.GetDim(0);
            int oldCols = x.GetDim(1);
            int newCols = oldCols - toDrop.Length;

            var newX = new FastTensor<float>(rows, newCols);
            var oldSpan = x.AsSpan();
            var newSpan = newX.AsSpan();

            var keepMask = new bool[oldCols];
            Array.Fill(keepMask, true);
            foreach (var idx in toDrop) keepMask[idx] = false;

            for (int r = 0; r < rows; r++)
            {
                int targetCol = 0;
                for (int c = 0; c < oldCols; c++)
                {
                    if (keepMask[c])
                    {
                        newSpan[r * newCols + targetCol] = oldSpan[r * oldCols + c];
                        targetCol++;
                    }
                }
            }
            return newX;
        }

        private float CalculatePearson(FastTensor<float> t, int colA, int colB)
        {
            int rows = t.GetDim(0);
            int cols = t.GetDim(1);
            var s = t.AsSpan(); //

            double sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;

            for (int r = 0; r < rows; r++)
            {
                float a = s[r * cols + colA];
                float b = s[r * cols + colB];
                sumA += a; sumB += b;
                sumAB += (double)a * b;
                sumA2 += (double)a * a;
                sumB2 += (double)b * b;
            }

            double num = (rows * sumAB) - (sumA * sumB);
            double den = Math.Sqrt((rows * sumA2 - sumA * sumA) * (rows * sumB2 - sumB * sumB));
            return den == 0 ? 0 : (float)(num / den);
        }

        public void Dispose() { }
    }
}