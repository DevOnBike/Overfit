using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Data
{
    public static class ModelSerializer
    {
        public static void SaveModel(string path, ConvLayer conv, LinearLayer fc)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);

            // 1. Zapisujemy Kernels (ConvLayer)
            SaveMatrix(bw, conv.Kernels.Data);

            // 2. Zapisujemy Weights i Bias (LinearLayer)
            SaveMatrix(bw, fc.Weights.Data);
            SaveMatrix(bw, fc.Biases.Data);
        
            // Możesz dodać metadane, np. wersję silnika lub datę
        }

        public static void LoadModel(string path, ConvLayer conv, LinearLayer fc)
        {
            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);

            // 1. Wczytujemy Kernels
            LoadMatrix(br, conv.Kernels.Data);

            // 2. Wczytujemy LinearLayer
            LoadMatrix(br, fc.Weights.Data);
            LoadMatrix(br, fc.Biases.Data);
        }

        private static void SaveMatrix(BinaryWriter bw, FastMatrix<double> matrix)
        {
            bw.Write(matrix.Rows);
            bw.Write(matrix.Cols);
            var data = matrix.AsSpan();
            for (var i = 0; i < data.Length; i++)
            {
                bw.Write(data[i]);
            }
        }

        private static void LoadMatrix(BinaryReader br, FastMatrix<double> matrix)
        {
            var rows = br.ReadInt32();
            var cols = br.ReadInt32();

            if (rows != matrix.Rows || cols != matrix.Cols)
            {
                throw new Exception($"Niezgodność wymiarów! Plik: {rows}x{cols}, Model: {matrix.Rows}x{matrix.Cols}");
            }

            var data = matrix.AsSpan();
            for (var i = 0; i < data.Length; i++)
            {
                data[i] = br.ReadDouble();
            }
        }
    }
}