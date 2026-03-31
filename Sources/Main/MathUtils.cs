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
        public static double NextGaussian()
        {
            var u1 = 1.0 - Rng.NextDouble(); // (0, 1]
            var u2 = 1.0 - Rng.NextDouble();

            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }
    }
}