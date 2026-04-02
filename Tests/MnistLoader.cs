using DevOnBike.Overfit.Core;

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

            // Zwracamy natywne Tensory zamiast Macierzy
            var images = new FastTensor<float>(maxSamples, 784);
            var labels = new FastTensor<float>(maxSamples, 10);

            for (var i = 0; i < maxSamples; i++)
            {
                var pixels = imgReader.ReadBytes(784);
                for (var j = 0; j < 784; j++)
                    images[i, j] = pixels[j] / 255.0f;

                int label = lblReader.ReadByte();
                labels[i, label] = 1.0f;
            }

            return (images, labels);
        }
    }
}