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
        /// The shape types multiply <b>unchecked</b>, and that is pinned rather than left to drift.
        ///
        /// <para>These products were made <c>checked</c> on 2026-08-03 and reverted on 2026-08-04: the
        /// dimensions come from the caller's own code rather than from a model file, so overflowing one
        /// means asking for a tensor larger than the machine can allocate, and the check measured ~0.11 ns
        /// per call for no benefit at this layer. The file-driven products — <c>GgufTensorInfo</c>,
        /// <c>OnnxTensor</c>, the ONNX graph importer — <b>are</b> checked and are pinned by
        /// <c>MalformedSizeAndPathTests</c>. This test exists so the distinction is a decision somebody
        /// wrote down, not an inconsistency somebody discovers and "fixes".</para>
        /// </summary>
        [Fact]
        public void ShapeArithmeticIsUncheckedByDesign()
        {
            var shape = new TensorShape(70_000, 70_000);

            // Wraps rather than throws. Reaching this needs a caller asking for 4.9 billion elements.
            var wrapped = shape.Size;

            Assert.NotEqual(70_000L * 70_000L, wrapped);

            Assert.Equal(2 * 3 * 4 * 5, new TensorShape(2, 3, 4, 5).Size);
            Assert.Equal(3 * 4, TensorStrides.Contiguous(new TensorShape(2, 3, 4)).S0);
        }

        /// <summary>
        /// <c>FastTensor.FromView</c>'s guard for a non-contiguous view of rank 1, 3 or 4 cannot be reached
        /// through the public API, and this test pins the reason rather than pretending to exercise it.
        ///
        /// <para><b>An earlier version of this test claimed to cover that case and did not</b> — it built a
        /// contiguous rank-3 view, so the guard never ran and the assertion only checked the pass-through
        /// path. Trying to write it properly is what surfaced the real fact: every public
        /// <see cref="TensorView{T}"/> constructor sets <c>isContiguous: true</c>, <c>Slice</c> and
        /// <c>Flatten</c> require contiguity and preserve it, and the single place that produces a
        /// non-contiguous view is <c>Transpose2D</c>, which throws unless the rank is 2. <b>A
        /// non-contiguous view therefore always has rank 2</b>, and the leak that guard prevents was
        /// latent and unconstructible rather than live.</para>
        ///
        /// <para>The guard stays — the copy loop below it indexes with two subscripts and would be wrong on
        /// any other rank — and this test pins the invariant that makes it unreachable. If somebody later
        /// adds a way to build a non-contiguous rank-3 view, this fails and points at the copy path that
        /// then needs writing for real.</para>
        /// </summary>
        [Fact]
        public void ANonContiguousViewIsAlwaysRankTwo()
        {
            var data = new float[24];

            for (var i = 0; i < data.Length; i++)
            {
                data[i] = i;
            }

            // Every constructed view is contiguous.
            Assert.True(new TensorView<float>(data, 24).IsContiguous);
            Assert.True(new TensorView<float>(data, 4, 6).IsContiguous);
            Assert.True(new TensorView<float>(data, 2, 3, 4).IsContiguous);
            Assert.True(new TensorView<float>(data, 1, 2, 3, 4).IsContiguous);

            // The one route to a non-contiguous view, and it is rank 2 by construction.
            var transposed = new TensorView<float>(data, 4, 6).Transpose2D();

            Assert.False(transposed.IsContiguous);
            Assert.Equal(2, transposed.Rank);

            // And it refuses every other rank, which is what keeps the invariant true.
            Assert.Throws<OverfitRuntimeException>(
                () => new TensorView<float>(data, 2, 3, 4).Transpose2D());
        }

        /// <summary>
        /// The supported path still materialises, so the guard added above it did not close the normal
        /// case — the half of a guard change that is easy to leave untested.
        /// </summary>
        [Fact]
        public void FromViewStillMaterialisesAContiguousView()
        {
            var data = new float[24];

            for (var i = 0; i < data.Length; i++)
            {
                data[i] = i;
            }

            using var materialized = FastTensor<float>.FromView(new TensorView<float>(data, 2, 3, 4));

            Assert.Equal(24, materialized.Size);
            Assert.Equal(23f, materialized.AsSpan()[23]);
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
