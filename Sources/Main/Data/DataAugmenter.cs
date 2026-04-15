// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Data
{
    public static class DataAugmenter
    {
        public static FastTensor<float> AugmentBatch(FastTensor<float> originalBatch, int width = 28, int height = 28)
        {
            var augmentedBatch = FastTensor<float>.SameShape(originalBatch, clearMemory: false);
            var rows = originalBatch.GetView().GetDim(0);

            Parallel.For(0, rows, body: i => {
                var inputRow = originalBatch.GetView().AsReadOnlySpan().Slice(i * width * height, width * height);
                var outputRow = augmentedBatch.GetView().AsSpan().Slice(i * width * height, width * height);

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

        private static void ShiftImage(ReadOnlySpan<float> input, Span<float> output, int w, int h, int sx, int sy)
        {
            output.Clear();

            for (var y = 0; y < h; y++)
            {
                for (var x = 0; x < w; x++)
                {
                    int nx = x + sx, ny = y + sy;
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h)
                    {
                        output[ny * w + nx] = input[y * w + x];
                    }
                }
            }
        }
    }
}