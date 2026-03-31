using DevOnBike.Overfit.Core;
namespace DevOnBike.Overfit.Tests
{
    public class FastMatrixTests
    {
        [Fact]
        public void Constructor_ValidDimensions_SetsPropertiesCorrectly()
        {
            // Act
            using var matrix = new FastMatrix<double>(3, 4);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(4, matrix.Cols);
        }

        [Fact]
        public void TensorSpan_ReturnsCorrectDimensions()
        {
            // Arrange
            using var matrix = new FastMatrix<double>(2, 5);

            // Act
            var tensor = matrix.AsTensor();

            // Assert
            Assert.Equal(2, tensor.Rank);
            Assert.Equal(2, tensor.Lengths[0]);
            Assert.Equal(5, tensor.Lengths[1]);
        }

        [Fact]
        public void Add_ValidMatrices_PerformsElementWiseAddition()
        {
            // Arrange
            using var m1 = new FastMatrix<double>(2, 2);
            using var m2 = new FastMatrix<double>(2, 2);

            m1[0, 0] = 1.0; m1[0, 1] = 2.0;
            m1[1, 0] = 3.0; m1[1, 1] = 4.0;

            m2[0, 0] = 10.0; m2[0, 1] = 20.0;
            m2[1, 0] = 30.0; m2[1, 1] = 40.0;

            // Act
            m1.Add(m2);

            // Assert
            Assert.Equal(11.0, m1[0, 0]);
            Assert.Equal(22.0, m1[0, 1]);
            Assert.Equal(33.0, m1[1, 0]);
            Assert.Equal(44.0, m1[1, 1]);
        }

        [Fact]
        public void MultiplyScalar_MultipliesAllElements()
        {
            // Arrange
            using var matrix = new FastMatrix<double>(2, 2);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = 4.0;

            // Act
            matrix.MultiplyScalar(5.0);

            // Assert
            Assert.Equal(5.0, matrix[0, 0]);
            Assert.Equal(10.0, matrix[0, 1]);
            Assert.Equal(15.0, matrix[1, 0]);
            Assert.Equal(20.0, matrix[1, 1]);
        }

        [Fact]
        public void Add_ShapeMismatch_ThrowsArgumentException()
        {
            // Arrange
            using var m1 = new FastMatrix<double>(2, 2);
            using var m2 = new FastMatrix<double>(3, 3); // Inny rozmiar

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => m1.Add(m2));
            Assert.Contains("Shape mismatch", ex.Message);
        }

        [Fact]
        public void AccessingMethodsAfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var matrix = new FastMatrix<double>(2, 2);
            matrix.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => matrix[0, 0]);
            Assert.Throws<ObjectDisposedException>(() => matrix.Row(0));
            Assert.Throws<ObjectDisposedException>(() => matrix.AsSpan());
            Assert.Throws<ObjectDisposedException>(() => matrix.AsTensor());
        }

        [Fact]
        public void SumOfSquares_ReturnsCorrectValue()
        {
            // Arrange
            using var matrix = new FastMatrix<double>(2, 2);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = 4.0;

            // Act
            var result = matrix.SumOfSquares();

            // Assert
            // 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
            Assert.Equal(30.0, result);
        }
    }
}