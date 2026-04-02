using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class FastMatrixTests
    {
        [Fact]
        public void Constructor_ValidDimensions_SetsPropertiesCorrectly()
        {
            using var matrix = new FloatFastMatrix(3, 4);

            Assert.Equal(3, matrix.Rows);
            Assert.Equal(4, matrix.Cols);
        }

        [Fact]
        public void TensorSpan_ReturnsCorrectDimensions()
        {
            using var matrix = new FloatFastMatrix(2, 5);
            var tensor = matrix.AsTensor();

            Assert.Equal(2, tensor.Rank);
            Assert.Equal(2, tensor.Lengths[0]);
            Assert.Equal(5, tensor.Lengths[1]);
        }

        [Fact]
        public void Add_ValidMatrices_PerformsElementWiseAddition()
        {
            using var m1 = new FloatFastMatrix(2, 2);
            using var m2 = new FloatFastMatrix(2, 2);

            m1[0, 0] = 1.0f; m1[0, 1] = 2.0f;
            m1[1, 0] = 3.0f; m1[1, 1] = 4.0f;

            m2[0, 0] = 10.0f; m2[0, 1] = 20.0f;
            m2[1, 0] = 30.0f; m2[1, 1] = 40.0f;

            m1.Add(m2);

            Assert.Equal(11.0f, m1[0, 0]);
            Assert.Equal(22.0f, m1[0, 1]);
            Assert.Equal(33.0f, m1[1, 0]);
            Assert.Equal(44.0f, m1[1, 1]);
        }

        [Fact]
        public void MultiplyScalar_MultipliesAllElements()
        {
            using var matrix = new FloatFastMatrix(2, 2);
            matrix[0, 0] = 1.0f; matrix[0, 1] = 2.0f;
            matrix[1, 0] = 3.0f; matrix[1, 1] = 4.0f;

            matrix.MultiplyScalar(5.0f);

            Assert.Equal(5.0f, matrix[0, 0]);
            Assert.Equal(10.0f, matrix[0, 1]);
            Assert.Equal(15.0f, matrix[1, 0]);
            Assert.Equal(20.0f, matrix[1, 1]);
        }

        [Fact]
        public void Add_ShapeMismatch_ThrowsArgumentException()
        {
            using var m1 = new FloatFastMatrix(2, 2);
            using var m2 = new FloatFastMatrix(3, 3);

            var ex = Assert.Throws<ArgumentException>(() => m1.Add(m2));
            Assert.Contains("Shape mismatch", ex.Message);
        }

        [Fact]
        public void AccessingMethodsAfterDispose_ThrowsObjectDisposedException()
        {
            var matrix = new FloatFastMatrix(2, 2);
            matrix.Dispose();

            Assert.Throws<ObjectDisposedException>(() => matrix[0, 0]);
            Assert.Throws<ObjectDisposedException>(() => matrix.Row(0));
            Assert.Throws<ObjectDisposedException>(() => matrix.AsSpan());
            Assert.Throws<ObjectDisposedException>(() => matrix.AsTensor());
        }

        [Fact]
        public void SumOfSquares_ReturnsCorrectValue()
        {
            using var matrix = new FloatFastMatrix(2, 2);
            matrix[0, 0] = 1.0f; matrix[0, 1] = 2.0f;
            matrix[1, 0] = 3.0f; matrix[1, 1] = 4.0f;

            var result = matrix.SumOfSquares();

            Assert.Equal(30.0f, result);
        }
    }
}