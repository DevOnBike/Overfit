// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using Benchmarks.Helpers;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// What the vectorised GELU is worth on a whole token, rather than on the activation alone.
    ///
    /// <para><b>Why this is a separate measurement and not an inference from the last one.</b>
    /// <see cref="GeluActivationBenchmark"/> measured the activation at 9.4-12.5x. That number says nothing
    /// about a token: the same activation could be 30% of a decode step or 0.3% of it, and multiplying a
    /// kernel speedup by an unmeasured share is how a real 10x becomes a claimed 10% that nobody can
    /// reproduce.</para>
    ///
    /// <para><b>GPT-2 small is the right subject for the upper bound.</b> Its FFN activation is plain GELU
    /// over the whole hidden layer, where a GeGLU model such as Gemma applies it only to the gate branch.
    /// If the share is negligible here it is negligible everywhere.</para>
    ///
    /// <para><b>The element count is 40,320 per generated token, NOT 36,864.</b> This comment claimed
    /// 36,864 — 12 layers x 3072 — until 2026-08-21, and a probe agreed with it because the probe divided
    /// the same way. <see cref="Decode"/> calls <c>Reset</c> first, so the decode also runs the <b>6 prompt
    /// positions</b>: 12 layers x 3072 units x <b>70</b> positions / 64 tokens = 40,320, or 13.125 calls per
    /// token. Anything expressed per element here rests on that denominator.</para>
    ///
    /// <para><b>Both arms run in ONE process.</b> A ratio taken across two process launches cannot separate
    /// the code change from the machine, and this repository has already had a 29% between-sitting drift on
    /// an untouched binary. <c>CachedFeedForwardBlock.VectorGelu</c> is flipped in the benchmark method
    /// itself, so the two arms share a JIT, a heap and a thermal state.</para>
    ///
    /// <para><b>The lever was proven live before this was written.</b> Making <c>ApplyGeLU</c> throw reddens
    /// all three GPT-2 KV-cache tests, so the decode path demonstrably reaches it. Without that check a flat
    /// 1.00 here would read as "GELU does not matter" when it would actually mean "the switch is not
    /// connected" — a reading this repository has produced before.</para>
    ///
    /// <para>Needs the GPT-2 small checkpoint: <c>OVERFIT_GPT2_DIR</c>, or <c>c:\gpt2</c>, or
    /// <c>test_fixtures/gpt2_small.bin</c>.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*GeluDecodeShare*"
    /// </summary>
    [Config(typeof(Config))]
    [MemoryDiagnoser]
    public class GeluDecodeShareBenchmark : IDisposable
    {
        private GPT1Model _model = null!;
        private CachedSlmInferenceEngine _engine = null!;
        private CachedSlmSession _session = null!;
        private int[] _prompt = null!;
        private SamplingOptions _sampling;
        private int _checksum;
        private float[] _canary = null!;
        private bool _disposed;

        [Params(64)]
        public int NewTokens { get; set; }

        private sealed class Config : ManualConfig
        {
            public Config()
            {
                // A 64-token decode runs in milliseconds, so the invocation count is left to
                // BenchmarkDotNet. The warmup is raised well above the shared config's five: five warmups
                // have already read one benchmark in this repository at 4x its steady-state value because
                // the method was still running tier-0 code.
                AddJob(Job.Default
                    .WithWarmupCount(10)
                    .WithIterationCount(15));
            }
        }

        [GlobalSetup]
        public void Setup()
        {
            var checkpointPath = BenchmarkModelPaths.ResolveGpt2SmallBinary();

            _model = new GPT1Model(Gpt2Config.Small);
            _model.Eval();

            using (var fs = File.OpenRead(checkpointPath))
            using (var br = new BinaryReader(fs))
            {
                _model.Load(br);
            }

            _engine = CachedSlmInferenceEngine.FromGpt1(_model);
            _session = _engine.CreateSession();
            _prompt = [464, 2003, 286, 3788, 2478, 318];
            _sampling = SamplingOptions.Greedy;
            _canary = new float[1_000_000];

            for (var i = 0; i < _canary.Length; i++)
            {
                _canary[i] = i * 1e-6f;
            }

            // Warm both arms, not just the one that happens to run first. Whichever arm BenchmarkDotNet
            // schedules second would otherwise pay another method's tiering on its first measured call.
            foreach (var vector in new[] { false, true })
            {
                CachedFeedForwardBlock.VectorGelu = vector;
                Decode();
            }

            CachedFeedForwardBlock.VectorGelu = true;
        }

        private void Decode()
        {
            _session.Reset(_prompt.AsSpan());

            for (var i = 0; i < NewTokens; i++)
            {
                _checksum ^= _session.GenerateNextToken(in _sampling);
            }
        }

        /// <summary>The scalar <c>MathF.Tanh</c> loop that shipped until 2026-08-19.</summary>
        [Benchmark(Baseline = true)]
        public void Decode_ScalarGelu()
        {
            CachedFeedForwardBlock.VectorGelu = false;

            Decode();
        }

        /// <summary>The vectorised identity that ships now.</summary>
        [Benchmark]
        public void Decode_VectorGelu()
        {
            CachedFeedForwardBlock.VectorGelu = true;

            Decode();
        }

        /// <summary>
        /// A fixed scalar-transcendental loop, untouched by the switch.
        ///
        /// <para><b>This is the canary the first two sittings did not have, and it is aimed at the specific
        /// suspect.</b> Across sittings the scalar decode arm moved 4.8% while the vectorised one moved
        /// 0.7%, which made the derived saving swing 2.3x — and the same signature appeared in
        /// <see cref="GeluActivationBenchmark"/>, where the canary was flat while the scalar arm jumped 13%.
        /// The standing hypothesis is that scalar transcendental throughput on this box is what moves, not
        /// the decode. A canary made of memory traffic could not tell that apart from anything else, so this
        /// one is made of the suspected quantity itself.</para>
        ///
        /// <para>Read it with <see cref="CanaryMemory"/>: both flat means the box held and a moving decode
        /// arm is real; this one moving alone points at clock or thermal state rather than at the code.</para>
        /// </summary>
        [Benchmark]
        public double CanaryScalarMath()
        {
            var total = 0.0;

            for (var i = 0; i < 1_000_000; i++)
            {
                total += MathF.Tanh((i & 1023) * 0.001f);
            }

            return total;
        }

        /// <summary>
        /// A fixed pass over memory, untouched by the switch. The other half of the pair: it moves when
        /// bandwidth or cache does, and stays put when only the scalar units are affected.
        /// </summary>
        [Benchmark]
        public float CanaryMemory()
        {
            var total = 0f;
            var buffer = _canary.AsSpan();

            for (var i = 0; i < buffer.Length; i++)
            {
                total += buffer[i];
            }

            return total;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            _session?.Dispose();
            _engine?.Dispose();
            _model?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}
