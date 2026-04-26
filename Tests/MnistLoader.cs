// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors.Core; // Zmieniono na Tensors.Core

namespace DevOnBike.Overfit.Tests
{
    public static class MnistLoader
    {
        public static (TensorStorage<float> images, TensorStorage<float> labels) Load(string imagesPath, string labelsPath, int maxSamples = 60000)
        {
            using var imgStream = File.OpenRead(imagesPath);
            using var lblStream = File.OpenRead(labelsPath);
            using var imgReader = new BinaryReader(imgStream);
            using var lblReader = new BinaryReader(lblStream);

            imgReader.ReadBytes(16);
            lblReader.ReadBytes(8);

            // DOD: Alokujemy płaską pamięć za pomocą TensorStorage
            var images = new TensorStorage<float>(maxSamples * 784, clearMemory: false);
            var labels = new TensorStorage<float>(maxSamples * 10, clearMemory: true);

            var imgSpan = images.AsSpan();
            var lblSpan = labels.AsSpan();

            for (var i = 0; i < maxSamples; i++)
            {
                var pixels = imgReader.ReadBytes(784);

                for (var j = 0; j < 784; j++)
                {
                    // Składamy indeks płasko: i * 784 + j
                    imgSpan[i * 784 + j] = pixels[j] / 255f;
                }

                var label = lblReader.ReadByte();
                // One-hot encoding na płaskiej strukturze
                lblSpan[i * 10 + label] = 1f;
            }

            return (images, labels);
        }
    }
}