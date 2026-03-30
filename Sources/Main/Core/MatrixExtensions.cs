namespace DevOnBike.Overfit.Core
{
    public static class MatrixExtensions
    {
        public static int ArgMax(this FastMatrix<double> matrix, int row = 0)
        {
            var span = matrix.Row(row);
            var maxIndex = 0;
            var maxValue = span[0];

            for (var i = 1; i < span.Length; i++)
            {
                if (span[i] > maxValue)
                {
                    maxValue = span[i];
                    maxIndex = i;
                }
            }
            
            return maxIndex;
        }
    }
}