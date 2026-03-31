using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public static class MnistLoader
    {
        public static (FastMatrix<double> images, FastMatrix<double> labels) Load(string imagesPath, string labelsPath, int maxSamples = 60000)
        {
            using var imgStream = File.OpenRead(imagesPath);
            using var lblStream = File.OpenRead(labelsPath);
            using var imgReader = new BinaryReader(imgStream);
            using var lblReader = new BinaryReader(lblStream);

            // Skip headers
            imgReader.ReadBytes(16);
            lblReader.ReadBytes(8);

            var images = new FastMatrix<double>(maxSamples, 784);
            var labels = new FastMatrix<double>(maxSamples, 10); // One-hot encoding

            for (var i = 0; i < maxSamples; i++)
            {
                // Read Image
                var pixels = imgReader.ReadBytes(784);
                for (var j = 0; j < 784; j++)
                    images[i, j] = pixels[j] / 255.0; // Normalizacja

                // Read Label & One-hot encode
                int label = lblReader.ReadByte();
                labels[i, label] = 1.0;
            }

            return (images, labels);
        }
    }
}