// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace Benchmarks
{
    /// <summary>
    /// What the guards added to <c>Sources/Main/Tensors</c> on 2026-08-03 cost: an overlap check on the
    /// raw-span <c>Add</c>, and <c>checked</c> on <see cref="TensorShape.Size"/> and
    /// <see cref="TensorStrides.Contiguous"/> — both <c>AggressiveInlining</c> members on tensor paths.
    ///
    /// <para><b>The first version of this file measured itself and is worth describing, because every
    /// mistake in it is one this repository has already paid for.</b> It reported the <i>guarded</i> Add as
    /// 0.71x — faster — which is backwards, and a backwards result means the benchmark before the runtime:
    /// the "before" arm was written as two loose calls while the shipped one is a single
    /// <c>AggressiveInlining</c> method, so it measured a call that did not inline. It reported
    /// <c>186 ns</c> identically for three different arms at all three sizes, because the shape was
    /// loop-invariant and the JIT hoisted the whole computation out. And it reported <c>Contiguous</c> at
    /// 2.47x, comparing a bare inline expression against a method that also evaluates <c>Rank</c> and
    /// switches on it — an A/B of two things at once.</para>
    ///
    /// <para><b>So every arm here is a copy of the shipped method differing in exactly one keyword</b>, both
    /// carrying <see cref="MethodImplOptions.AggressiveInlining"/>, and the inputs vary per iteration so
    /// nothing can be hoisted.</para>
    ///
    /// <para><b>Read the canary first.</b> <see cref="CanaryScale"/> and <see cref="CanaryScaleAgain"/> are
    /// the same untouched call. If their ratio is not ~1.00 the box moved during the run and the rest is
    /// noise.</para>
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    public class TensorGuardCostBenchmark
    {
        private const int Repeats = 1000;

        private float[] _left = [];
        private float[] _right = [];
        private float[] _destination = [];

        /// <summary>Varying shapes, so no product is loop-invariant and none can be folded or hoisted.</summary>
        private TensorShape[] _shapes = [];

        [Params(64, 4096, 65536)]
        public int Elements
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            _left = new float[Elements];
            _right = new float[Elements];
            _destination = new float[Elements];

            for (var i = 0; i < Elements; i++)
            {
                _left[i] = i * 0.5f;
                _right[i] = i * 0.25f;
            }

            _shapes = new TensorShape[Repeats];

            for (var i = 0; i < Repeats; i++)
            {
                _shapes[i] = new TensorShape(2 + (i % 5), 3 + (i % 7), 4 + (i % 3), 1 + (i % 11));
            }
        }

        // ---- the shipped methods, copied, differing in one thing each ----

        /// <summary>The raw-span <c>Add</c> as it was before 2026-08-03: length validation only.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AddBefore(
            ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination)
        {
            TensorKernelGuards.ValidateSameLengthAndDestination(left, right, destination);
            TensorPrimitives.Add(left, right, destination);
        }

        /// <inheritdoc cref="TensorShape.Size"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int SizeBefore(in TensorShape shape)
        {
            return shape.D0 * shape.D1 * shape.D2 * shape.D3;
        }

        /// <inheritdoc cref="TensorStrides.Contiguous"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TensorStrides ContiguousBefore(TensorShape shape)
        {
            return shape.Rank switch
            {
                1 => new TensorStrides(1),
                2 => new TensorStrides(shape.D1, 1),
                3 => new TensorStrides(shape.D1 * shape.D2, shape.D2, 1),
                4 => new TensorStrides(shape.D1 * shape.D2 * shape.D3, shape.D2 * shape.D3, shape.D3, 1),
                _ => throw new InvalidOperationException($"Unsupported rank: {shape.Rank}"),
            };
        }

        // ---- Add: with and without the overlap guard ----

        [Benchmark(Baseline = true)]
        public void AddBeforeGuard()
        {
            AddBefore(_left, _right, _destination);
        }

        [Benchmark]
        public void AddWithGuard()
        {
            TensorKernels.Add(_left, _right, _destination);
        }

        // ---- checked arithmetic, one keyword apart, over varying shapes ----

        [Benchmark]
        public int SizeUnchecked()
        {
            var total = 0;
            var shapes = _shapes;

            for (var i = 0; i < shapes.Length; i++)
            {
                total += SizeBefore(shapes[i]);
            }

            return total;
        }

        [Benchmark]
        public int SizeChecked()
        {
            var total = 0;
            var shapes = _shapes;

            for (var i = 0; i < shapes.Length; i++)
            {
                total += shapes[i].Size;
            }

            return total;
        }

        [Benchmark]
        public int ContiguousUnchecked()
        {
            var total = 0;
            var shapes = _shapes;

            for (var i = 0; i < shapes.Length; i++)
            {
                total += ContiguousBefore(shapes[i]).S0;
            }

            return total;
        }

        [Benchmark]
        public int ContiguousChecked()
        {
            var total = 0;
            var shapes = _shapes;

            for (var i = 0; i < shapes.Length; i++)
            {
                total += TensorStrides.Contiguous(shapes[i]).S0;
            }

            return total;
        }

        // ---- canary: the same untouched call, twice ----

        [Benchmark]
        public void CanaryScale()
        {
            TensorKernels.Scale(_left, 1.5f, _destination);
        }

        /// <inheritdoc cref="CanaryScale"/>
        [Benchmark]
        public void CanaryScaleAgain()
        {
            TensorKernels.Scale(_left, 1.5f, _destination);
        }
    }
}
