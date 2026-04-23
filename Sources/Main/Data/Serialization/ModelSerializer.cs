// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors.Core; // Wpinamy Core!

namespace DevOnBike.Overfit.Data.Serialization
{
    public static class ModelSerializer
    {
        public static void SaveModel(string path, ConvLayer conv, LinearLayer fc)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);

            SaveTensor(bw, conv.Kernels.DataView);
            SaveTensor(bw, fc.Weights.DataView);
            SaveTensor(bw, fc.Bias.DataView); // Zmieniono Biases na Bias (zgodnie z LinearLayer)
        }

        public static void LoadModel(string path, ConvLayer conv, LinearLayer fc)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Model file not found: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);

            LoadTensor(br, conv.Kernels.DataView);
            LoadTensor(br, fc.Weights.DataView);
            LoadTensor(br, fc.Bias.DataView); // Zmieniono Biases na Bias
        }

        private static void SaveTensor(BinaryWriter bw, TensorSpan<float> view) // TensorSpan zamiast TensorView
        {
            bw.Write(view.Rank);

            for (var i = 0; i < view.Rank; i++)
            {
                bw.Write(view.GetDim(i));
            }

            var data = view.AsReadOnlySpan();

            for (var i = 0; i < data.Length; i++)
            {
                bw.Write(data[i]);
            }
        }

        private static void LoadTensor(BinaryReader br, TensorSpan<float> view) // TensorSpan zamiast TensorView
        {
            var rank = br.ReadInt32();
            var fileShape = new int[rank];
            for (var i = 0; i < rank; i++)
            {
                fileShape[i] = br.ReadInt32();
            }

            if (rank != view.Rank)
            {
                throw new Exception($"Tensor rank mismatch! File: {rank}, Model: {view.Rank}");
            }

            for (var i = 0; i < rank; i++)
            {
                if (fileShape[i] != view.GetDim(i))
                {
                    throw new Exception($"Dimension mismatch at index {i}! File: {fileShape[i]}, Model: {view.GetDim(i)}");
                }
            }

            var data = view.AsSpan();
            for (var i = 0; i < data.Length; i++)
            {
                data[i] = br.ReadSingle();
            }
        }
    }
}