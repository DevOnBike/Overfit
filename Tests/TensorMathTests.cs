namespace DevOnBike.Overfit.Tests
{
    public class TensorMathTests
    {
        private const int Precision = 5;

        [Fact]
        public void BroadcastRowVector_CreatesCorrectVirtualMatrix()
        {
            // Arrange
            using var rowVector = new FastBuffer<double>(3);
            rowVector[0] = 1.5;
            rowVector[1] = 2.5;
            rowVector[2] = 3.5;

            var targetRows = 4;

            // Act
            var broadcastedView = TensorMath.BroadcastRowVector(rowVector.AsSpan(), targetRows);

            // Assert Metadata
            Assert.Equal(4, broadcastedView.Rows);
            Assert.Equal(3, broadcastedView.Cols);
            Assert.Equal(0, broadcastedView.RowStride); // The magic zero-stride!
            Assert.Equal(1, broadcastedView.ColStride);

            // Assert Data (Every row should return the exact same values)
            for (var r = 0; r < targetRows; r++)
            {
                Assert.Equal(1.5, broadcastedView[r, 0], Precision);
                Assert.Equal(2.5, broadcastedView[r, 1], Precision);
                Assert.Equal(3.5, broadcastedView[r, 2], Precision);
            }
        }

        [Fact]
        public void Add_StandardMatrices_AddsElementWise()
        {
            // Arrange
            using var m1 = new FastMatrix<double>(2, 2);
            using var m2 = new FastMatrix<double>(2, 2);
            using var result = new FastMatrix<double>(2, 2);

            m1[0, 0] = 1.0; m1[0, 1] = 2.0;
            m1[1, 0] = 3.0; m1[1, 1] = 4.0;

            m2[0, 0] = 10.0; m2[0, 1] = 20.0;
            m2[1, 0] = 30.0; m2[1, 1] = 40.0;

            // Act
            TensorMath.Add(m1.AsView(), m2.AsView(), result.AsView());

            // Assert
            Assert.Equal(11.0, result[0, 0], Precision);
            Assert.Equal(22.0, result[0, 1], Precision);
            Assert.Equal(33.0, result[1, 0], Precision);
            Assert.Equal(44.0, result[1, 1], Precision);
        }

        [Fact]
        public void Add_WithBroadcastedBias_AddsBiasToEveryRow()
        {
            // Arrange
            // 1. Activations batch (3 rows, 2 columns)
            using var activations = new FastMatrix<double>(3, 2);
            activations[0, 0] = 1.0; activations[0, 1] = 2.0;
            activations[1, 0] = 3.0; activations[1, 1] = 4.0;
            activations[2, 0] = 5.0; activations[2, 1] = 6.0;

            // 2. Bias vector (1D array of 2 elements)
            using var bias = new FastBuffer<double>(2);
            bias[0] = 10.0; 
            bias[1] = 20.0;

            // 3. Result matrix
            using var result = new FastMatrix<double>(3, 2);

            // Act: Broadcast the bias and add it to activations
            var broadcastedBias = TensorMath.BroadcastRowVector(bias.AsSpan(), targetRows: 3);
            TensorMath.Add(activations.AsView(), broadcastedBias, result.AsView());

            // Assert
            // Row 0: [1, 2] + [10, 20] = [11, 22]
            Assert.Equal(11.0, result[0, 0], Precision);
            Assert.Equal(22.0, result[0, 1], Precision);

            // Row 1: [3, 4] + [10, 20] = [13, 24]
            Assert.Equal(13.0, result[1, 0], Precision);
            Assert.Equal(24.0, result[1, 1], Precision);

            // Row 2: [5, 6] + [10, 20] = [15, 26]
            Assert.Equal(15.0, result[2, 0], Precision);
            Assert.Equal(26.0, result[2, 1], Precision);
        }

        [Fact]
        public void Add_ShapeMismatch_ThrowsArgumentException()
        {
            // Arrange
            using var m1 = new FastMatrix<double>(2, 2);
            using var m2 = new FastMatrix<double>(3, 3); // Mismatch!
            using var result = new FastMatrix<double>(2, 2);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => 
                TensorMath.Add(m1.AsView(), m2.AsView(), result.AsView()));
            
            Assert.Contains("Shape mismatch", ex.Message);
        }
        
        [Fact]
        public void MatMul_ValidMatrices_CalculatesCorrectDotProducts()
        {
            // Arrange
            // Macierz A (2x3)
            using var A = new FastMatrix<double>(2, 3);
            A[0, 0] = 1; A[0, 1] = 2; A[0, 2] = 3;
            A[1, 0] = 4; A[1, 1] = 5; A[1, 2] = 6;

            // Macierz B (3x2)
            using var B = new FastMatrix<double>(3, 2);
            B[0, 0] = 7;  B[0, 1] = 8;
            B[1, 0] = 9;  B[1, 1] = 10;
            B[2, 0] = 11; B[2, 1] = 12;

            // Oczekiwany wynik C = A * B (2x2)
            // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
            // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
            // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
            // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
            using var C = new FastMatrix<double>(2, 2);

            // Act
            TensorMath.MatMul(A.AsView(), B.AsView(), C.AsView());

            // Assert
            Assert.Equal(58.0, C[0, 0], Precision);
            Assert.Equal(64.0, C[0, 1], Precision);
            Assert.Equal(139.0, C[1, 0], Precision);
            Assert.Equal(154.0, C[1, 1], Precision);
        }

        [Fact]
        public void MatMul_IdentityMatrix_ReturnsOriginalMatrix()
        {
            // Arrange
            using var A = new FastMatrix<double>(2, 2);
            A[0, 0] = 42.5; A[0, 1] = -13.3;
            A[1, 0] = 7.1;  A[1, 1] = 0.0;

            using var I = new FastMatrix<double>(2, 2);
            I[0, 0] = 1; I[0, 1] = 0;
            I[1, 0] = 0; I[1, 1] = 1;

            using var C = new FastMatrix<double>(2, 2);

            // Act
            TensorMath.MatMul(A.AsView(), I.AsView(), C.AsView());

            // Assert
            Assert.Equal(42.5, C[0, 0], Precision);
            Assert.Equal(-13.3, C[0, 1], Precision);
            Assert.Equal(7.1, C[1, 0], Precision);
            Assert.Equal(0.0, C[1, 1], Precision);
        }

        [Fact]
        public void MatMul_ShapeMismatch_ThrowsArgumentException()
        {
            // Arrange
            using var A = new FastMatrix<double>(2, 3); // A ma 3 kolumny
            using var B = new FastMatrix<double>(4, 2); // B ma 4 wiersze (Mismatch!)
            using var C = new FastMatrix<double>(2, 2);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => 
                TensorMath.MatMul(A.AsView(), B.AsView(), C.AsView()));

            Assert.Contains("Shape mismatch", ex.Message);
        }

        [Fact]
        public void MatMul_ResultMatrixShapeMismatch_ThrowsArgumentException()
        {
            // Arrange
            using var A = new FastMatrix<double>(2, 3);
            using var B = new FastMatrix<double>(3, 2);
            // C powinno być 2x2, dajemy 3x3
            using var C = new FastMatrix<double>(3, 3); 

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => 
                TensorMath.MatMul(A.AsView(), B.AsView(), C.AsView()));

            Assert.Contains("Shape mismatch", ex.Message);
        }
    }
}