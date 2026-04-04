namespace DevOnBike.Overfit.Core
{
    public static class FastTensorExtensions
    {
        public static FastTensor<float> Randomize(this FastTensor<float> tensor, float scale = 0.01f)
        {
            var span = tensor.AsSpan();

            for (var i = 0; i < span.Length; i++)
            {
                // Inicjalizacja Xavier/Glorot lub prosta losowa
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
