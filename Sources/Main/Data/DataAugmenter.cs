using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Data
{
    public static class DataAugmenter
    {
        public static FastTensor<float> AugmentBatch(FastTensor<float> originalBatch, int width = 28, int height = 28)
        {
            var batchSize = originalBatch.Shape[0];
            var augmented = new FastTensor<float>(originalBatch.Shape);

            Parallel.For(0, batchSize, i =>
            {
                var input = originalBatch.AsSpan().Slice(i * width * height, width * height);
                var output = augmented.AsSpan().Slice(i * width * height, width * height);

                if (Random.Shared.NextSingle() > 0.5f)
                {
                    ShiftImage(input, output, width, height, Random.Shared.Next(-2, 3), Random.Shared.Next(-2, 3));
                }
                else input.CopyTo(output);
            });

            return augmented;
        }

        private static void ShiftImage(ReadOnlySpan<float> input, Span<float> output, int w, int h, int sx, int sy)
        {
            output.Clear();
            for (var y = 0; y < h; y++)
            {
                for (var x = 0; x < w; x++)
                {
                    int nx = x + sx, ny = y + sy;
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h)
                        output[ny * w + nx] = input[y * w + x];
                }
            }
        }
    }
}