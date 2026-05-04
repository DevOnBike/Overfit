// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Evaluators;

namespace DevOnBike.Overfit.Tests
{
    public sealed class ParallelPopulationEvaluatorTests
    {
        [Fact]
        public void Evaluate_CallsEvaluatorExactlyOncePerGenome()
        {
            var counter = new CallCounter();

            using var evaluator = new ParallelPopulationEvaluator<Scratch>(
                counter,
                () => new Scratch());

            var population = new float[32 * 4];
            var fitness = new float[32];

            evaluator.Evaluate(population, fitness, populationSize: 32, parameterCount: 4);

            Assert.Equal(32, counter.TotalCalls);
        }

        [Fact]
        public void Evaluate_WritesSumOfParametersAsFitness()
        {
            using var evaluator = new ParallelPopulationEvaluator<Scratch>(
                new SumEvaluator(),
                () => new Scratch());

            const int populationSize = 16;
            const int parameterCount = 4;

            var population = new float[populationSize * parameterCount];
            for (var i = 0; i < population.Length; i++)
            {
                population[i] = i;
            }

            var fitness = new float[populationSize];

            evaluator.Evaluate(population, fitness, populationSize, parameterCount);

            // Row i of the population is floats [i*4 .. i*4+3], summing to i*16 + 6.
            for (var i = 0; i < populationSize; i++)
            {
                var expected = (i * parameterCount * parameterCount) + 6f;
                Assert.Equal(expected, fitness[i], 4);
            }
        }

        [Fact]
        public void Evaluate_OnEmptyPopulation_IsNoOp()
        {
            using var evaluator = new ParallelPopulationEvaluator<Scratch>(
                new SumEvaluator(),
                () => new Scratch());

            evaluator.Evaluate(ReadOnlySpan<float>.Empty, Span<float>.Empty, populationSize: 0, parameterCount: 4);
            // Nothing to assert — the test passes if no exception is thrown.
        }

        [Fact]
        public void Evaluate_ThrowsOnMismatchedPopulationLength()
        {
            using var evaluator = new ParallelPopulationEvaluator<Scratch>(
                new SumEvaluator(),
                () => new Scratch());

            var population = new float[10];
            var fitness = new float[4];

            Assert.Throws<ArgumentException>(
                () => evaluator.Evaluate(population, fitness, populationSize: 4, parameterCount: 4));
        }

        [Fact]
        public void Evaluate_ThrowsOnMismatchedFitnessLength()
        {
            using var evaluator = new ParallelPopulationEvaluator<Scratch>(
                new SumEvaluator(),
                () => new Scratch());

            var population = new float[16];
            var fitness = new float[5];

            Assert.Throws<ArgumentException>(
                () => evaluator.Evaluate(population, fitness, populationSize: 4, parameterCount: 4));
        }

        [Fact]
        public void ContextFactory_IsInvokedAtMostOncePerWorkerThread()
        {
            // For the 16-physical-core machine this test is developed on, ProcessorCount is
            // typically 32. Across 256 evaluations we still expect at most ProcessorCount
            // context instances to be created — ThreadLocal reuses values from prior calls.
            var factoryInvocations = 0;

            var factory = () =>
            {
                Interlocked.Increment(ref factoryInvocations);
                return new Scratch();
            };

            using var evaluator = new ParallelPopulationEvaluator<Scratch>(
                new SumEvaluator(),
                factory);

            var population = new float[256 * 4];
            var fitness = new float[256];

            evaluator.Evaluate(population, fitness, populationSize: 256, parameterCount: 4);

            Assert.InRange(factoryInvocations, 1, Environment.ProcessorCount);
        }

        [Fact]
        public void ContextFactory_IsReusedAcrossSubsequentEvaluateCalls()
        {
            // Invoking Evaluate multiple times must not re-create the contexts. After a
            // warmup call (which may spin up up to ProcessorCount workers), subsequent calls
            // should allocate zero new contexts regardless of how many times they run.
            var factoryInvocations = 0;

            var factory = () =>
            {
                Interlocked.Increment(ref factoryInvocations);
                return new Scratch();
            };

            using var evaluator = new ParallelPopulationEvaluator<Scratch>(
                new SumEvaluator(),
                factory);

            var population = new float[64 * 4];
            var fitness = new float[64];

            evaluator.Evaluate(population, fitness, 64, 4);
            var afterFirstCall = factoryInvocations;

            // Second call across the same workload: factory must NOT fire again
            // for any already-touched thread.
            for (var run = 0; run < 5; run++)
            {
                evaluator.Evaluate(population, fitness, 64, 4);
            }

            // Some additional threads may join the pool during subsequent runs (the thread
            // pool is allowed to grow), but the count must not grow unboundedly per call.
            Assert.True(
                factoryInvocations - afterFirstCall <= Environment.ProcessorCount,
                $"Factory ran {factoryInvocations - afterFirstCall} extra times after first call, " +
                $"expected at most {Environment.ProcessorCount}.");
        }

        [Fact]
        public void ContextDispose_IsCalledOnDispose_ForEveryLiveContext()
        {
            var disposeInvocations = 0;

            var evaluator = new ParallelPopulationEvaluator<Scratch>(
                new SumEvaluator(),
                () => new Scratch(),
                contextDispose: _ => Interlocked.Increment(ref disposeInvocations));

            var population = new float[64 * 4];
            var fitness = new float[64];

            evaluator.Evaluate(population, fitness, 64, 4);

            // Capture the set of contexts that got created during evaluation.
            // Dispose must run the user-supplied cleanup on each of them.
            evaluator.Dispose();

            Assert.InRange(disposeInvocations, 1, Environment.ProcessorCount);
        }

        [Fact]
        public void Evaluate_AfterDispose_Throws()
        {
            var evaluator = new ParallelPopulationEvaluator<Scratch>(
                new SumEvaluator(),
                () => new Scratch());

            evaluator.Dispose();

            var population = new float[4];
            var fitness = new float[1];

            Assert.Throws<ObjectDisposedException>(
                () => evaluator.Evaluate(population, fitness, 1, 4));
        }

        [Fact(Skip = "Flaky test: Pomiary czasu (ticks) Parallel vs Sequential są niestabilne w zależności od obciążenia systemu/liczby rdzeni.")]
        public void Evaluate_RunsFasterThanSequential_ForExpensivePerGenomeWork()
        {
            // Sanity check that parallel dispatch is actually wired up. We inject an
            // evaluator with artificial per-genome work (a spin loop) and compare against
            // a sequential reference. On any multi-core machine, parallel should beat
            // sequential by a clear margin. On a single-core machine or under extreme
            // contention the assertion is skipped.
            if (Environment.ProcessorCount < 2)
            {
                return;
            }

            const int populationSize = 64;
            const int parameterCount = 4;
            const int spinIterations = 50_000;

            var population = new float[populationSize * parameterCount];
            var fitness = new float[populationSize];

            // Sequential baseline: run the spin-evaluator inline.
            var spinEvaluator = new SpinEvaluator(spinIterations);
            var sequentialStopwatch = Stopwatch.StartNew();

            for (var i = 0; i < populationSize; i++)
            {
                var parameters = new ReadOnlySpan<float>(population, i * parameterCount, parameterCount);
                var scratch = new Scratch();
                fitness[i] = spinEvaluator.Evaluate(parameters, ref scratch);
            }

            sequentialStopwatch.Stop();

            // Parallel: full Parallel.For dispatch.
            using var evaluator = new ParallelPopulationEvaluator<Scratch>(
                spinEvaluator,
                () => new Scratch());

            // Warmup to amortize the thread-pool spin-up.
            evaluator.Evaluate(population, fitness, populationSize, parameterCount);

            var parallelStopwatch = Stopwatch.StartNew();
            evaluator.Evaluate(population, fitness, populationSize, parameterCount);
            parallelStopwatch.Stop();

            // Allow significant headroom: on loaded CI runners, cold pool, and modest
            // parallelism the actual speedup varies. We just need to see that parallel
            // was meaningfully faster — not that it hit a theoretical maximum.
            Assert.True(
                parallelStopwatch.ElapsedTicks * 2 < sequentialStopwatch.ElapsedTicks,
                $"Parallel ({parallelStopwatch.ElapsedTicks} ticks) was not at least 2x faster " +
                $"than sequential ({sequentialStopwatch.ElapsedTicks} ticks).");
        }

        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private sealed class Scratch
        {
        }

        private sealed class SumEvaluator : ICandidateEvaluator<Scratch>
        {
            public float Evaluate(ReadOnlySpan<float> parameters, ref Scratch context)
            {
                var sum = 0f;

                for (var i = 0; i < parameters.Length; i++)
                {
                    sum += parameters[i];
                }

                return sum;
            }
        }

        private sealed class CallCounter : ICandidateEvaluator<Scratch>
        {
            private int _totalCalls;

            public int TotalCalls => _totalCalls;

            public float Evaluate(ReadOnlySpan<float> parameters, ref Scratch context)
            {
                Interlocked.Increment(ref _totalCalls);
                return 0f;
            }
        }

        /// <summary>
        ///     Stand-in for a real fitness function: performs a fixed amount of busy work
        ///     per call. Used only by the parallel-vs-sequential timing sanity check.
        /// </summary>
        private sealed class SpinEvaluator : ICandidateEvaluator<Scratch>
        {
            private readonly int _iterations;

            public SpinEvaluator(int iterations)
            {
                _iterations = iterations;
            }

            public float Evaluate(ReadOnlySpan<float> parameters, ref Scratch context)
            {
                // Busy-work loop that the JIT cannot eliminate (the final result depends on
                // the input parameters and the loop index).
                var acc = 0f;

                for (var i = 0; i < _iterations; i++)
                {
                    acc += MathF.Sin(i + (parameters.Length > 0 ? parameters[0] : 0f));
                }

                return acc;
            }
        }
    }
}