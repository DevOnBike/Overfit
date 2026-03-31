using DevOnBike.Overfit.Core;
namespace DevOnBike.Overfit.Tests
{
    public class CholeskyMultivariateGaussianLogicTests
    {
        private const int Precision = 5; // Dokładność dla porównań zmiennoprzecinkowych

        [Fact]
        public void ValidateInputs_ValidInputs_DoesNotThrow()
        {
            // Arrange
            var mean = new double[] { 1.0, 2.0 };
            using var covariance = new FastMatrix<double>(2, 2);

            // Act & Assert
            var ex = Record.Exception(() => CholeskyMultivariateGaussianLogic.ValidateInputs(mean, covariance));
            Assert.Null(ex);
        }

        [Fact]
        public void ValidateInputs_EmptyMean_ThrowsArgumentException()
        {
            // Arrange
            var mean = Array.Empty<double>();
            using var covariance = new FastMatrix<double>(2, 2);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => CholeskyMultivariateGaussianLogic.ValidateInputs(mean, covariance));
            Assert.Contains("empty", ex.Message);
        }

        [Fact]
        public void ValidateInputs_DimensionMismatch_ThrowsArgumentException()
        {
            // Arrange
            var mean = new double[] { 1.0, 2.0 };
            using var covariance = new FastMatrix<double>(3, 3); // Mismatch!

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => CholeskyMultivariateGaussianLogic.ValidateInputs(mean, covariance));
            Assert.Contains("must be 2x2", ex.Message);
        }

        [Fact]
        public void DecomposeCholesky_ValidPositiveDefiniteMatrix_ReturnsCorrectL()
        {
            // Arrange
            using var covariance = new FastMatrix<double>(3, 3);
            covariance[0, 0] = 4; covariance[0, 1] = 12; covariance[0, 2] = -16;
            covariance[1, 0] = 12; covariance[1, 1] = 37; covariance[1, 2] = -43;
            covariance[2, 0] = -16; covariance[2, 1] = -43; covariance[2, 2] = 98;

            // Expected L:
            // [  2,   0,   0 ]
            // [  6,   1,   0 ]
            // [ -8,   5,   3 ]

            // Act
            using var L = CholeskyMultivariateGaussianLogic.DecomposeCholesky(covariance);

            // Assert
            Assert.Equal(2.0, L[0, 0], Precision); Assert.Equal(0.0, L[0, 1]); Assert.Equal(0.0, L[0, 2]);
            Assert.Equal(6.0, L[1, 0], Precision); Assert.Equal(1.0, L[1, 1], Precision); Assert.Equal(0.0, L[1, 2]);
            Assert.Equal(-8.0, L[2, 0], Precision); Assert.Equal(5.0, L[2, 1], Precision); Assert.Equal(3.0, L[2, 2], Precision);
        }

        [Fact]
        public void DecomposeCholesky_NotPositiveDefinite_ThrowsArgumentExceptionAndCleansUp()
        {
            // Arrange
            using var covariance = new FastMatrix<double>(2, 2);
            covariance[0, 0] = 4; covariance[0, 1] = 12;
            covariance[1, 0] = 12; covariance[1, 1] = 9; // Sprawia, że macierz nie jest dodatnio określona

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
            {
                // Chociaż metoda wewnątrz rzuca wyjątek, powinna poprawnie wywołać L.Dispose(),
                // co zabezpiecza naszą pulę ArrayPool przed wyciekiem pamięci.
                _ = CholeskyMultivariateGaussianLogic.DecomposeCholesky(covariance);
            });

            Assert.Contains("not positive-definite", ex.Message);
        }

        [Fact]
        public void CalculateLogNormConstant_ValidIdentityMatrix_ReturnsCorrectConstant()
        {
            // Arrange
            var dimensions = 2;
            using var L = new FastMatrix<double>(dimensions, dimensions);
            L[0, 0] = 1.0; L[0, 1] = 0.0;
            L[1, 0] = 0.0; L[1, 1] = 1.0;

            // Constant dla macierzy identycznościowej: -0.5 * 2 * ln(2PI) - 0
            var expectedConstant = -1.0 * Math.Log(2.0 * Math.PI);

            // Act
            var result = CholeskyMultivariateGaussianLogic.CalculateLogNormConstant(dimensions, L);

            // Assert
            Assert.Equal(expectedConstant, result, Precision);
        }

        [Fact]
        public void LogProbabilityDensity_ValidInputs_ReturnsCorrectDensity()
        {
            // Arrange
            var dimensions = 2;
            var observation = new double[] { 0.5, 0.5 };
            var mean = new double[] { 0.5, 0.5 }; // Obserwacja == Średnia -> odległość Mahalanobisa = 0

            using var L = new FastMatrix<double>(dimensions, dimensions);
            L[0, 0] = 1.0; L[1, 1] = 1.0; // Macierz identycznościowa

            var logNormConst = CholeskyMultivariateGaussianLogic.CalculateLogNormConstant(dimensions, L);

            // Skoro distance = 0, LogProbabilityDensity powinno być równe logNormConst
            var expectedLogProb = logNormConst - 0.0;

            // Act
            var result = CholeskyMultivariateGaussianLogic.LogProbabilityDensity(observation, mean, L, logNormConst);

            // Assert
            Assert.Equal(expectedLogProb, result, Precision);
        }

        [Fact]
        public void LogProbabilityDensity_DimensionMismatch_ThrowsArgumentException()
        {
            // Arrange
            var observation = new double[] { 1.0, 2.0, 3.0 }; // Długość 3
            var mean = new double[] { 1.0, 2.0 }; // Długość 2
            using var L = new FastMatrix<double>(2, 2);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                CholeskyMultivariateGaussianLogic.LogProbabilityDensity(observation, mean, L, 0.0));

            Assert.Contains("must have length 2", ex.Message);
        }
    }
}