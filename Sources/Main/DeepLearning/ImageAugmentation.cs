// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Simple train-time image augmentation over channel-major <c>[C, H, W]</c> float buffers — random
    /// translation and additive Gaussian noise. Apply to each training image before it enters the graph
    /// (NOT at inference); produces fresh variants each epoch so the model generalises instead of
    /// memorising. No autograd — these operate on raw pixel data.
    /// </summary>
    public static class ImageAugmentation
    {
        /// <summary>
        /// Translates the image by a random integer offset in <c>[−maxShift, maxShift]²</c> (same offset
        /// for all channels), zero-filling exposed borders, writing into <paramref name="destination"/>.
        /// </summary>
        public static void RandomShift(
            ReadOnlySpan<float> source, Span<float> destination, int channels, int height, int width, int maxShift, Random rng)
        {
            ArgumentNullException.ThrowIfNull(rng);
            ArgumentOutOfRangeException.ThrowIfNegative(maxShift);
            var size = channels * height * width;
            if (source.Length < size || destination.Length < size)
            {
                throw new ArgumentException("source/destination smaller than channels*height*width.");
            }

            destination.Slice(0, size).Clear();
            var dy = rng.Next(-maxShift, maxShift + 1);
            var dx = rng.Next(-maxShift, maxShift + 1);

            for (var c = 0; c < channels; c++)
            {
                var plane = c * height * width;
                for (var y = 0; y < height; y++)
                {
                    var sy = y - dy;
                    if (sy < 0 || sy >= height)
                    {
                        continue;
                    }
                    for (var x = 0; x < width; x++)
                    {
                        var sx = x - dx;
                        if (sx < 0 || sx >= width)
                        {
                            continue;
                        }
                        destination[plane + y * width + x] = source[plane + sy * width + sx];
                    }
                }
            }
        }

        /// <summary>Adds zero-mean Gaussian noise (standard deviation <paramref name="sigma"/>) in place.</summary>
        public static void AddGaussianNoise(Span<float> image, float sigma, Random rng)
        {
            ArgumentNullException.ThrowIfNull(rng);
            if (sigma <= 0f)
            {
                return;
            }
            for (var i = 0; i < image.Length; i++)
            {
                image[i] += sigma * NextGaussian(rng);
            }
        }

        // Box–Muller standard normal from the supplied RNG (reproducible).
        private static float NextGaussian(Random rng)
        {
            var u1 = 1.0 - rng.NextDouble();   // (0, 1]
            var u2 = rng.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
    }
}
