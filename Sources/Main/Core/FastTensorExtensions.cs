// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Core
{
    public static class FastTensorExtensions
    {
        /// <summary>
        /// Randomizes tensor elements using a simple scale. 
        /// </summary>
        public static FastTensor<float> Randomize(this FastTensor<float> tensor, float scale = 0.01f)
        {
            var span = tensor.AsSpan();
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = (Random.Shared.NextSingle() * 2f - 1f) * scale;
            }
            return tensor;
        }

        public static FastTensor<float> Fill(this FastTensor<float> tensor, float value)
        {
            tensor.AsSpan().Fill(value);
            return tensor;
        }
    }
}