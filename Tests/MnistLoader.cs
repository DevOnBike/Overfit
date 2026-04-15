// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Tests
{
    public static class MnistLoader
    {
        public static (FastTensor<float> images, FastTensor<float> labels) Load(string imagesPath, string labelsPath, int maxSamples = 60000)
        {
            using var imgStream = File.OpenRead(imagesPath);
            using var lblStream = File.OpenRead(labelsPath);
            using var imgReader = new BinaryReader(imgStream);
            using var lblReader = new BinaryReader(lblStream);

            imgReader.ReadBytes(16);
            lblReader.ReadBytes(8);

            // Zwracamy natywne Tensory. Obrazy nie muszą być czyszczone, etykiety tak (one-hot).
            var images = new FastTensor<float>(maxSamples, 784, clearMemory: false);
            var labels = new FastTensor<float>(maxSamples, 10, clearMemory: true);

            var imgSpan = images.GetView().AsSpan();
            var lblSpan = labels.GetView().AsSpan();

            for (var i = 0; i < maxSamples; i++)
            {
                var pixels = imgReader.ReadBytes(784);
                for (var j = 0; j < 784; j++)
                {
                    imgSpan[i * 784 + j] = pixels[j] / 255.0f;
                }

                var label = lblReader.ReadByte();
                lblSpan[i * 10 + label] = 1.0f;
            }

            return (images, labels);
        }
    }
}