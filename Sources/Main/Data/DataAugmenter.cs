using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Data
{
    public static class DataAugmenter
    {
        public static FastMatrix<float> AugmentBatch(FastMatrix<float> originalBatch, int width = 28, int height = 28)
        {
            var augmentedBatch = new FastMatrix<float>(originalBatch.Rows, originalBatch.Cols);

            Parallel.For(0, originalBatch.Rows, i =>
            {
                var inputRow = originalBatch.Row(i);
                var outputRow = augmentedBatch.Row(i);

                if (Random.Shared.NextSingle() > 0.5f)
                {
                    var shiftX = Random.Shared.Next(-2, 3);
                    var shiftY = Random.Shared.Next(-2, 3);

                    ShiftImage(inputRow, outputRow, width, height, shiftX, shiftY);
                }
                else
                {
                    inputRow.CopyTo(outputRow);
                }
            });

            return augmentedBatch;
        }

        private static void ShiftImage(ReadOnlySpan<float> input, Span<float> output, int width, int height, int shiftX, int shiftY)
        {
            output.Clear();

            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    var newX = x + shiftX;
                    var newY = y + shiftY;

                    if (newX >= 0 && newX < width && newY >= 0 && newY < height)
                    {
                        var oldIndex = y * width + x;
                        var newIndex = newY * width + newX;

                        output[newIndex] = input[oldIndex];
                    }
                }
            }
        }
    }
}