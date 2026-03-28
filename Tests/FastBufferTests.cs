namespace DevOnBike.Overfit.Tests
{
    public class FastBufferTests
    {
        [Fact]
        public void Constructor_ValidLength_SetsLengthAndClearsMemory()
        {
            // Act
            using var buffer = new FastBuffer<int>(10);

            // Assert
            Assert.Equal(10, buffer.Length);
            
            for (var i = 0; i < buffer.Length; i++)
            {
                Assert.Equal(0, buffer[i]); // Pamięć z ArrayPool musi być wyzerowana
            }
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-5)]
        public void Constructor_InvalidLength_ThrowsArgumentOutOfRangeException(int length)
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new FastBuffer<int>(length));
        }

        [Fact]
        public void Indexer_SetAndGet_WorksCorrectly()
        {
            // Arrange
            using var buffer = new FastBuffer<double>(5);
            
            // Act
            buffer[2] = 42.5;

            // Assert
            Assert.Equal(42.5, buffer[2]);
        }

        [Fact]
        public void AsSpan_ReturnsSpanOfExactLength()
        {
            // Arrange
            var length = 7;
            using var buffer = new FastBuffer<int>(length);

            // Act
            var span = buffer.AsSpan();

            // Assert
            Assert.Equal(length, span.Length);
        }

        [Fact]
        public void Dispose_CanBeCalledMultipleTimesWithoutException()
        {
            // Arrange
            var buffer = new FastBuffer<int>(5);

            // Act
            var exception = Record.Exception(() =>
            {
                buffer.Dispose();
                buffer.Dispose(); // Podwójne wywołanie zabezpieczone przez Interlocked
            });

            // Assert
            Assert.Null(exception);
        }

        [Fact]
        public void AccessingMethodsAfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var buffer = new FastBuffer<int>(5);
            buffer.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => buffer[0]);
            Assert.Throws<ObjectDisposedException>(() => buffer.AsSpan());
            Assert.Throws<ObjectDisposedException>(() => buffer.AsReadOnlySpan());
            Assert.Throws<ObjectDisposedException>(() => buffer.Clear());
        }
    }
}