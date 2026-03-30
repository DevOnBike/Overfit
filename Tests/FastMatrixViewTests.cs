using DevOnBike.Overfit.Core;
namespace DevOnBike.Overfit.Tests
{
    public class FastMatrixViewTests
    {
        // Pomocnicza metoda do tworzenia ciągłego widoku 2x3 dla testów
        // Reprezentacja wizualna macierzy:
        // [ 1, 2, 3 ]
        // [ 4, 5, 6 ]
        private double[] CreateTestData() => new double[] { 1, 2, 3, 4, 5, 6 };

        [Fact]
        public void Indexer_ReadAndWrite_WorksOnUnderlyingSpan()
        {
            // Arrange
            var data = CreateTestData();
            var span = data.AsSpan();
            var view = new FastMatrixView<double>(span, rows: 2, cols: 3, rowStride: 3, colStride: 1, offset: 0);

            // Act & Assert (Read)
            Assert.Equal(1, view[0, 0]);
            Assert.Equal(3, view[0, 2]);
            Assert.Equal(4, view[1, 0]);
            Assert.Equal(6, view[1, 2]);

            // Act (Write)
            view[1, 1] = 99; // Zmieniamy wartość '5' na '99'

            // Assert (Underlying memory changed)
            Assert.Equal(99, data[4]);
        }

        [Fact]
        public void IsContiguous_CalculatesCorrectly()
        {
            // Arrange
            var span = CreateTestData().AsSpan();
            
            // Continuous View
            var view = new FastMatrixView<double>(span, rows: 2, cols: 3, rowStride: 3, colStride: 1, offset: 0);
            
            // Non-Continuous View (e.g., reading every second element)
            var nonContiguousView = new FastMatrixView<double>(span, rows: 2, cols: 2, rowStride: 3, colStride: 2, offset: 0);

            // Assert
            Assert.True(view.IsContiguous);
            Assert.False(nonContiguousView.IsContiguous);
        }

        [Fact]
        public void Transpose_SwapsDimensionsAndStrides_AndReadsCorrectly()
        {
            // Arrange
            var span = CreateTestData().AsSpan();
            var view = new FastMatrixView<double>(span, rows: 2, cols: 3, rowStride: 3, colStride: 1, offset: 0);

            // Act
            var transposed = view.Transpose();

            // Assert metadata
            Assert.Equal(3, transposed.Rows);
            Assert.Equal(2, transposed.Cols);
            Assert.Equal(1, transposed.RowStride);
            Assert.Equal(3, transposed.ColStride);
            Assert.Equal(0, transposed.Offset);

            // Transponowany widok już nie jest ciągły w pamięci (IsContiguous == false)
            Assert.False(transposed.IsContiguous);

            // Assert data mapping
            // view[1, 0] = 4, so transposed[0, 1] should be 4
            Assert.Equal(view[1, 0], transposed[0, 1]);
            Assert.Equal(view[0, 2], transposed[2, 0]); // 3 == 3
        }

        [Fact]
        public void Slice_ValidBounds_UpdatesOffsetAndDimensions()
        {
            // Arrange
            var span = CreateTestData().AsSpan();
            var view = new FastMatrixView<double>(span, rows: 2, cols: 3, rowStride: 3, colStride: 1, offset: 0);

            // Act: Wycinamy podmacierz 1x2 z prawego dolnego rogu
            // [ x, x, x ]
            // [ x, 5, 6 ] <- startRow: 1, startCol: 1
            var slice = view.Slice(1, 1, rows: 1, cols: 2);

            // Assert metadata
            Assert.Equal(1, slice.Rows);
            Assert.Equal(2, slice.Cols);
            // Offset powinien wskazywać na indeks 4 (wartość '5') w oryginalnej tablicy
            Assert.Equal(4, slice.Offset); 

            // Assert data
            Assert.Equal(5, slice[0, 0]);
            Assert.Equal(6, slice[0, 1]);
        }

        [Fact]
        public void Slice_ExceedsBounds_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var span = CreateTestData().AsSpan();
            var view = new FastMatrixView<double>(span, rows: 2, cols: 3, rowStride: 3, colStride: 1, offset: 0);

            // Act & Assert
            // Używamy try-catch zamiast Assert.Throws, ponieważ 'view' to 'ref struct' 
            // i nie może zostać przekazane do wnętrza wyrażenia lambda () => ...
            ArgumentOutOfRangeException? expectedException = null;
            try
            {
                view.Slice(1, 2, rows: 2, cols: 2); // Wychodzi poza wymiar
            }
            catch (ArgumentOutOfRangeException ex)
            {
                expectedException = ex;
            }

            Assert.NotNull(expectedException);
            Assert.Contains("Slice exceeds view dimensions", expectedException.Message);
        }

        [Fact]
        public void Row_ContiguousMemory_ReturnsCorrectSpan()
        {
            // Arrange
            var span = CreateTestData().AsSpan();
            var view = new FastMatrixView<double>(span, rows: 2, cols: 3, rowStride: 3, colStride: 1, offset: 0);

            // Act
            var rowSpan = view.Row(1); // Drugi wiersz: [ 4, 5, 6 ]

            // Assert
            Assert.Equal(3, rowSpan.Length);
            Assert.Equal(4, rowSpan[0]);
            Assert.Equal(5, rowSpan[1]);
            Assert.Equal(6, rowSpan[2]);
        }

        [Fact]
        public void Row_NonContiguousMemory_ThrowsInvalidOperationException()
        {
            // Arrange
            var span = CreateTestData().AsSpan();
            var view = new FastMatrixView<double>(span, rows: 2, cols: 3, rowStride: 3, colStride: 1, offset: 0);
            
            // Tworzymy transpozycję. ColStride staje się > 1, więc pamięć wiersza nie jest ciągła.
            var transposed = view.Transpose();

            // Act & Assert
            InvalidOperationException? expectedException = null;
            try
            {
                transposed.Row(0); // Próba pobrania ciągłego wiersza z transponowanej macierzy
            }
            catch (InvalidOperationException ex)
            {
                expectedException = ex;
            }

            Assert.NotNull(expectedException);
            Assert.Contains("Cannot return a contiguous Span", expectedException.Message);
        }
    }
}