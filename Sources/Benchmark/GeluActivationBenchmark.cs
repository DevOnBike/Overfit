// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// The GELU gate in the decode FFN, which is still scalar.
    ///
    /// <para><b>Why this one.</b> <c>CachedFeedForwardBlock.ApplySiLU</c> carries the note that
    /// <i>"the scalar path's per-element MathF.Exp was the bottleneck"</i> and is vectorised through
    /// <c>TensorPrimitives.Sigmoid</c>. <c>ApplyGeLU</c>, forty lines below it, still runs one
    /// <c>MathF.Tanh</c> per element and says so: <i>"Vectorization can be done later if this becomes a
    /// bottleneck."</i> The twin path was measured to be a bottleneck; this one has never been measured at
    /// all. That is what this benchmark is for.</para>
    ///
    /// <para><b>Who pays it.</b> Every GeGLU model (Gemma) on the gate branch, and every plain-GELU model
    /// (GPT-2) on the whole hidden layer, once per token.</para>
    ///
    /// <para><b>The identity the fast arm rests on, stated so it can be checked rather than trusted.</b>
    /// <c>tanh(z) = (1 - e^-2z) / (1 + e^-2z)</c>, so <c>0.5 * (1 + tanh(z)) = 1 / (1 + e^-2z) = sigmoid(2z)</c>
    /// exactly. The tanh approximation of GELU is therefore <c>x * sigmoid(2z)</c> with the same <c>z</c>, and
    /// the rewrite is an algebraic identity rather than a second approximation — it reuses the very
    /// <c>TensorPrimitives.Sigmoid</c> the SiLU path above was already fixed with. Parity against the scalar
    /// form is pinned by a test, not by this benchmark.</para>
    ///
    /// <para><b>What would refute the hypothesis.</b> The vectorised arms make three passes over the buffer
    /// where the scalar one makes a single pass with an expensive transcendental inside it. At these sizes
    /// the buffer sits in L1 or L2, so if <c>MathF.Tanh</c> is cheaper than two extra passes the fast arms
    /// lose — and that is a real possibility, not a formality. <see cref="Canary"/> is an untouched path run
    /// in the same session; if it moves between sittings the box moved and no ratio here is readable.</para>
    ///
    /// <para><b>Source to destination, not in place.</b> The shipped method works in place, but an in-place
    /// benchmark either re-applies GELU to its own output — which drifts toward zero and lands in denormal
    /// timing — or pays a restoring copy inside the measured window that dilutes the ratio. Reading a
    /// pristine source is the same arithmetic and no cheaper for either arm.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*GeluActivation*"
    /// </summary>
    [Config(typeof(Config))]
    [MemoryDiagnoser]
    public class GeluActivationBenchmark
    {
        private const float SqrtTwoOverPi = 0.7978845608028654f;
        private const float Coeff = 0.044715f;

        private float[] _source = null!;
        private float[] _destination = null!;
        private float[] _scratch = null!;

        /// <summary>GPT-2 small, Qwen2.5-3B and Gemma-2 9B hidden widths, in that order.</summary>
        [Params(3072, 11008, 14336)]
        public int Width { get; set; }

        private sealed class Config : ManualConfig
        {
            public Config()
            {
                // Deliberately NOT the shared BenchmarkConfig. That one pins InvocationCount(1), which is
                // right for a millisecond-scale model call and wrong here: this operation runs in
                // microseconds, so one invocation per iteration measures the timer and the tiering, not the
                // kernel. Letting BenchmarkDotNet choose the invocation count is the correct shape at this
                // scale.
                AddJob(Job.Default
                    .WithWarmupCount(10)
                    .WithIterationCount(15));
            }
        }

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260819);

            _source = new float[Width];
            _destination = new float[Width];
            _scratch = new float[Width];

            for (var i = 0; i < Width; i++)
            {
                // Pre-activation values in a transformer FFN are roughly zero-centred and of order one.
                // The range matters: GELU is nearly linear for large positive x and nearly zero for large
                // negative x, and both tails are cheaper to compute than the transition around zero.
                _source[i] = (float)((rng.NextDouble() * 8.0) - 4.0);
            }
        }

        /// <summary>Exactly what <c>CachedFeedForwardBlock.ApplyGeLU</c> ships today.</summary>
        [Benchmark(Baseline = true)]
        public void ScalarTanh()
        {
            var source = _source.AsSpan();
            var destination = _destination.AsSpan();

            for (var i = 0; i < source.Length; i++)
            {
                var x = source[i];
                var x3 = x * x * x;
                var inner = SqrtTwoOverPi * (x + (Coeff * x3));

                destination[i] = 0.5f * x * (1f + MathF.Tanh(inner));
            }
        }

        /// <summary>The algebraic identity: <c>x * sigmoid(2z)</c>, reusing the SiLU path's own primitive.</summary>
        [Benchmark]
        public void VectorSigmoid()
        {
            var source = _source.AsSpan();
            var destination = _destination.AsSpan();
            var scratch = _scratch.AsSpan();

            for (var i = 0; i < source.Length; i++)
            {
                var x = source[i];
                var x3 = x * x * x;

                scratch[i] = 2f * SqrtTwoOverPi * (x + (Coeff * x3));
            }

            TensorPrimitives.Sigmoid(scratch, scratch);
            TensorPrimitives.Multiply(source, scratch, destination);
        }

        /// <summary>The direct translation, kept so the identity is not credited with a win it did not earn.</summary>
        [Benchmark]
        public void VectorTanh()
        {
            var source = _source.AsSpan();
            var destination = _destination.AsSpan();
            var scratch = _scratch.AsSpan();

            for (var i = 0; i < source.Length; i++)
            {
                var x = source[i];
                var x3 = x * x * x;

                scratch[i] = SqrtTwoOverPi * (x + (Coeff * x3));
            }

            TensorPrimitives.Tanh(scratch, scratch);
            TensorPrimitives.Add(scratch, 1f, scratch);
            TensorPrimitives.Multiply(source, scratch, destination);
            TensorPrimitives.Multiply(destination, 0.5f, destination);
        }

        /// <summary>
        /// The method that actually ships, called through <c>InternalsVisibleTo</c>.
        ///
        /// <para><b>This arm exists because the arms above measure a COPY of the shipped shape, not the
        /// shipped shape.</b> That distinction has already cost this repository a whole sweep once, when a
        /// harness kept its own stale binary and an entire measurement described the previous library. The
        /// method works in place, so the source is copied first; the copy is inside the measured window and
        /// costs about what <see cref="Canary"/> costs, i.e. under 3% here.</para>
        /// </summary>
        [Benchmark]
        public void ShippedApplyGeLU()
        {
            _source.AsSpan().CopyTo(_destination);

            CachedFeedForwardBlock.ApplyGeLU(_destination.AsSpan());
        }

        /// <summary>
        /// An untouched path over the same buffer. It is not a candidate — it exists so that a ratio read in
        /// one sitting can be compared against another. If this arm moves, the box moved.
        /// </summary>
        [Benchmark]
        public void Canary()
        {
            TensorPrimitives.Multiply(_source.AsSpan(), 1f, _destination.AsSpan());
        }
    }
}
