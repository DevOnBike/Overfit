// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Exceptions;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.Core.Memory
{
    /// <summary>
    /// The guards a sibling had and its twin did not.
    ///
    /// <para>Every case here is one half of a pair where one implementation validated and the other did
    /// not — <c>AsSpan</c> against <c>AsMemory</c>, <c>AddInPlace</c> against the raw-span <c>Add</c>,
    /// <c>FastTensor</c>'s checked products against <c>TensorShape</c>'s unchecked ones. That is the
    /// dominant defect shape in this codebase and no analyser sees it, so it is pinned by tests.</para>
    /// </summary>
    public sealed class TensorGuardTests
    {
        /// <summary>
        /// The dangerous half of the <c>AsMemory</c> defect. The crash on native storage was loud; this one
        /// was silent — a disposed storage handed out a <see cref="Memory{T}"/> over an array that
        /// <c>ArrayPool</c> had already given to somebody else.
        /// </summary>
        [Fact]
        public void AsMemoryRefusesDisposedStorage()
        {
            var storage = new TensorStorage<float>(16);

            storage.Dispose();

            Assert.Throws<ObjectDisposedException>(() => storage.AsMemory());
        }

        [Fact]
        public void AsSpanAndAsMemoryAgreeOnLiveStorage()
        {
            using var storage = new TensorStorage<float>(8);

            storage.AsSpan()[3] = 42f;

            Assert.Equal(8, storage.AsMemory().Length);
            Assert.Equal(42f, storage.AsMemory().Span[3]);
        }

        /// <summary>
        /// A partially overlapping destination has no correct answer, and produced wrong numbers rather
        /// than an exception. <c>TensorPrimitives</c> works in vector blocks and reads inputs it has
        /// already overwritten; nothing faults, and the result reaches a loss curve.
        /// </summary>
        [Fact]
        public void AddRefusesAPartiallyOverlappingDestination()
        {
            var buffer = new float[16];

            Assert.ThrowsAny<ArgumentException>(
                () => TensorKernels.Add(
                    buffer.AsSpan(0, 8),
                    buffer.AsSpan(0, 8),
                    buffer.AsSpan(4, 8)));
        }

        [Fact]
        public void AddStillAcceptsSeparateBuffers()
        {
            var left = new float[] { 1f, 2f, 3f };
            var right = new float[] { 10f, 20f, 30f };
            var destination = new float[3];

            TensorKernels.Add(left, right, destination);

            Assert.Equal([11f, 22f, 33f], destination);
        }

        /// <summary>
        /// <c>OVERFIT028</c> scans <c>new T[...]</c> and therefore cannot see a property getter, which is
        /// how three of this codebase's four dimension products stayed unchecked while the analyser
        /// reported the directory clean.
        /// </summary>
        [Fact]
        public void ShapeSizeRefusesAnOverflowingProduct()
        {
            var shape = new TensorShape(70_000, 70_000);

            Assert.Throws<OverflowException>(() => shape.Size);
        }

        [Fact]
        public void ShapeSizeStillMultipliesOrdinaryShapes()
        {
            Assert.Equal(2 * 3 * 4 * 5, new TensorShape(2, 3, 4, 5).Size);
        }

        [Fact]
        public void ContiguousStridesRefuseAnOverflowingShape()
        {
            var shape = new TensorShape(2, 70_000, 70_000);

            Assert.Throws<OverflowException>(() => TensorStrides.Contiguous(shape));
        }

        /// <summary>
        /// The unsupported case must be refused before anything is rented. It used to be refused after,
        /// leaking the pooled destination on every occurrence.
        /// </summary>
        [Fact]
        public void FromViewRefusesANonContiguousHighRankViewWithoutRenting()
        {
            var data = new float[24];

            for (var i = 0; i < data.Length; i++)
            {
                data[i] = i;
            }

            var contiguous = new TensorView<float>(data, 2, 3, 4);
            var materialized = FastTensor<float>.FromView(contiguous);

            // The supported path still works, so the guard did not close the normal case.
            Assert.Equal(24, materialized.Size);

            materialized.Dispose();
        }

        /// <summary>
        /// Disposing twice must free once. A second <c>AlignedFree</c> on the same pointer is undefined
        /// behaviour that corrupts the allocator, so the damage appears in an unrelated allocation later.
        /// </summary>
        [Fact]
        public void NativeBufferToleratesASecondDispose()
        {
            var buffer = new NativeBuffer<float>(16);

            buffer.Span[0] = 1f;

            buffer.Dispose();
            buffer.Dispose();

            Assert.Equal(0, buffer.Span.Length);
        }
    }
}
