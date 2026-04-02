namespace DevOnBike.Overfit
{
    public static class MathUtils
    {
        [ThreadStatic]
        private static Random _rng;

        private static Random Rng => _rng ??= new Random(Guid.NewGuid().GetHashCode());

        /// <summary>
        /// Zwraca liczbę losową z rozkładu normalnego N(0, 1) używając transformacji Box-Mullera.
        /// </summary>
        public static float NextGaussian()
        {
            const float twoPi = 2.0f * MathF.PI;
            
            var u1 = 1.0f - Rng.NextSingle(); // (0, 1]
            var u2 = 1.0f - Rng.NextSingle();

            return MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Sin(twoPi * u2);
        }
    }
}