// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tensors
{
    public static class FastTensorExtensions
    {
        /// <summary>
        /// Fills the tensor with uniform values in <c>[-scale, scale)</c>, drawn from
        /// <see cref="Maths.MathUtils"/> so <see cref="Maths.MathUtils.SetSeed"/> reaches it.
        ///
        /// <para>This drew from <c>Random.Shared</c>, which no seed can control — a model initialised
        /// through here could not be made reproducible by any caller. See the note on
        /// <see cref="Maths.MathUtils.NextSingle"/> for what that cost.</para>
        /// </summary>
        public static FastTensor<float> Randomize(this FastTensor<float> tensor, float scale = 0.01f)
        {
            var span = tensor.GetView().AsSpan();

            for (var i = 0; i < span.Length; i++)
            {
                span[i] = (Maths.MathUtils.NextSingle() * 2f - 1f) * scale;
            }

            return tensor;
        }

        public static FastTensor<float> Fill(this FastTensor<float> tensor, float value)
        {
            tensor.GetView().AsSpan().Fill(value);

            return tensor;
        }
    }
}