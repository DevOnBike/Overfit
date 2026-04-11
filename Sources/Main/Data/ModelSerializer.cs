// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Data
{
    /// <summary>
    ///     Utility class for binary serialization of model parameters.
    ///     Handles persisting and restoring tensor data for convolutional and linear layers.
    /// </summary>
    public static class ModelSerializer
    {
        /// <summary>
        ///     Saves the model weights and biases to a binary file.
        /// </summary>
        public static void SaveModel(string path, ConvLayer conv, LinearLayer fc)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);

            SaveTensor(bw, conv.Kernels.Data);
            SaveTensor(bw, fc.Weights.Data);
            SaveTensor(bw, fc.Biases.Data);
        }

        /// <summary>
        ///     Loads model weights and biases from a binary file.
        ///     Throws an exception if the file structure does not match the expected tensor shapes.
        /// </summary>
        public static void LoadModel(string path, ConvLayer conv, LinearLayer fc)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Model file not found: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);

            LoadTensor(br, conv.Kernels.Data);
            LoadTensor(br, fc.Weights.Data);
            LoadTensor(br, fc.Biases.Data);
        }

        /// <summary>
        ///     Serializes a FastTensor to the stream using the format: [Rank][Dimensions][RawData].
        /// </summary>
        private static void SaveTensor(BinaryWriter bw, FastTensor<float> tensor)
        {
            bw.Write(tensor.Shape.Length);

            for (var i = 0; i < tensor.Shape.Length; i++)
            {
                bw.Write(tensor.Shape[i]);
            }

            var data = tensor.AsSpan();

            for (var i = 0; i < data.Length; i++)
            {
                bw.Write(data[i]);
            }
        }

        /// <summary>
        ///     Deserializes data from the stream into an existing FastTensor.
        ///     Performs strict validation of rank and dimensions before loading data.
        /// </summary>
        private static void LoadTensor(BinaryReader br, FastTensor<float> tensor)
        {
            var rank = br.ReadInt32();
            var fileShape = new int[rank];
            for (var i = 0; i < rank; i++)
            {
                fileShape[i] = br.ReadInt32();
            }

            if (rank != tensor.Shape.Length)
            {
                throw new Exception($"Tensor rank mismatch! File: {rank}, Model: {tensor.Shape.Length}");
            }

            for (var i = 0; i < rank; i++)
            {
                if (fileShape[i] != tensor.Shape[i])
                {
                    throw new Exception($"Dimension mismatch at index {i}! File: {fileShape[i]}, Model: {tensor.Shape[i]}");
                }
            }

            // Load data directly into the tensor's Span
            var data = tensor.AsSpan();
            for (var i = 0; i < data.Length; i++)
            {
                data[i] = br.ReadSingle();
            }
        }
    }
}