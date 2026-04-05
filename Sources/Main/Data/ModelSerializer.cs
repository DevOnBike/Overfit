// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

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
            SaveTensor(bw, conv.Kernels.Data);

            // 2. Zapisujemy Weights i Bias (LinearLayer)
            SaveTensor(bw, fc.Weights.Data);
            SaveTensor(bw, fc.Biases.Data);
        }

        public static void LoadModel(string path, ConvLayer conv, LinearLayer fc)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Brak pliku modelu: {path}");

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);

            // 1. Wczytujemy Kernels
            LoadTensor(br, conv.Kernels.Data);

            // 2. Wczytujemy LinearLayer
            LoadTensor(br, fc.Weights.Data);
            LoadTensor(br, fc.Biases.Data);
        }

        private static void SaveTensor(BinaryWriter bw, FastTensor<float> tensor)
        {
            // Zapisujemy rangę (liczbę wymiarów)
            bw.Write(tensor.Shape.Length);

            // Zapisujemy poszczególne wymiary Shape
            for (var i = 0; i < tensor.Shape.Length; i++)
            {
                bw.Write(tensor.Shape[i]);
            }

            // Zapisujemy surowe dane ze Spana
            var data = tensor.AsSpan();
            for (var i = 0; i < data.Length; i++)
            {
                bw.Write(data[i]);
            }
        }

        private static void LoadTensor(BinaryReader br, FastTensor<float> tensor)
        {
            // Odczytujemy rangę i kształt z pliku
            var rank = br.ReadInt32();
            var fileShape = new int[rank];
            for (var i = 0; i < rank; i++)
            {
                fileShape[i] = br.ReadInt32();
            }

            // Weryfikacja zgodności wymiarów
            if (rank != tensor.Shape.Length)
            {
                throw new Exception($"Niezgodność rangi tensora! Plik: {rank}, Model: {tensor.Shape.Length}");
            }

            for (var i = 0; i < rank; i++)
            {
                if (fileShape[i] != tensor.Shape[i])
                {
                    throw new Exception($"Niezgodność wymiaru {i}! Plik: {fileShape[i]}, Model: {tensor.Shape[i]}");
                }
            }

            // Wczytujemy dane bezpośrednio do Spana tensora
            var data = tensor.AsSpan();
            for (var i = 0; i < data.Length; i++)
            {
                data[i] = br.ReadSingle();
            }
        }
    }
}