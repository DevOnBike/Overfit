namespace DevOnBike.Overfit.Core
{
    public class DoubleFastMatrix : FastMatrix<double>
    {
        public DoubleFastMatrix(int rows, int cols) : base(rows, cols)
        {
        }

        /// <inheritdoc/>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                // 1. Zwolnij WŁASNE zasoby managed
            }

            // 2. Wywołaj base — zwalnia _data z FastMatrix<T>
            // ZAWSZE jako ostatni krok — base może ustawić _disposed = 1
            base.Dispose(disposing);
        }
    }
}