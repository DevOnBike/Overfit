// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Tests
{
    public class BufferTests
    {
        [Fact]
        public void PooledBuffer_ShouldHaveExactSpanSize_EvenIfPoolReturnsLargerArray()
        {
            // Arrange & Act
            using var buffer = new PooledBuffer<float>(1000);

            // Assert
            Assert.Equal(1000, buffer.Span.Length);
        }

        [Fact]
        public void PooledBuffer_ClearMemory_ShouldZeroOutData()
        {
            // Arrange & Act
            using var buffer = new PooledBuffer<int>(50, clearMemory: true);

            // Assert
            foreach (var val in buffer.Span)
            {
                Assert.Equal(0, val);
            }
        }

        [Fact]
        public void PooledBuffer_ShouldNotThrowOnDispose()
        {
            // Ominięcie użycia lambdy w Assert.Throws dla ref struct
            var exception = Record.Exception(() =>
            {
                using var buffer = new PooledBuffer<double>(100);
                buffer.Span[0] = 1.0;
            });

            Assert.Null(exception);
        }

        // UWAGA: Testy dla NativeBuffer wymagają włączonego <AllowUnsafeBlocks>true</AllowUnsafeBlocks> w pliku .csproj testów
        [Fact]
        public unsafe void NativeBuffer_ShouldAllocateAndFreeCorrectly()
        {
            // Arrange
            var exception = Record.Exception(() =>
            {
                using var nativeBuffer = new NativeBuffer<float>(256);

                // Act
                nativeBuffer.Span[0] = 42.0f;
                nativeBuffer.Span[255] = 24.0f;

                // Assert
                Assert.Equal(42.0f, nativeBuffer.Span[0]);
                Assert.Equal(24.0f, nativeBuffer.Span[255]);
                Assert.Equal(256, nativeBuffer.Span.Length);
            });

            Assert.Null(exception);
        }

        [Fact]
        public unsafe void NativeBuffer_ClearMemory_ShouldZeroOutData()
        {
            // Arrange & Act
            using var nativeBuffer = new NativeBuffer<int>(100, clearMemory: true);

            // Assert
            for (var i = 0; i < 100; i++)
            {
                Assert.Equal(0, nativeBuffer.Span[i]);
            }
        }
    }
}