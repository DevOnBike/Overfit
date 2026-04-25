// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Runtime;
using DevOnBike.Overfit.Evolutionary.Storage;

namespace DevOnBike.Overfit.Tests
{
    public sealed class MapElitesTests
    {
        [Fact]
        public void FirstIteration_FillsAtLeastOneArchiveCell()
        {
            using var archive = CreateArchive(parameterCount: 8);
            var evaluator = new SimpleDescriptorEvaluator();
            using var map = new MapElites<DummyContext>(
                parameterCount: 8,
                batchSize: 64,
                archive: archive,
                evaluator: evaluator,
                seed: 123,
                mutationSigma: 0.05f,
                initialMin: -1f,
                initialMax: 1f,
                randomInjectionProbability: 0.1f);

            var ctx = new DummyContext();
            var metrics = map.RunIteration(ref ctx);

            Assert.True(metrics.InsertedNewCells > 0);
            Assert.True(metrics.OccupiedCells > 0);
            Assert.True(metrics.Coverage > 0f);
            Assert.True(map.HasBestEvaluated);
            Assert.False(map.GetBestEvaluatedParameters().IsEmpty);
        }

        [Fact]
        public void SameSeed_ProducesSameTrajectory()
        {
            using var archiveA = CreateArchive(parameterCount: 8);
            using var archiveB = CreateArchive(parameterCount: 8);

            var evaluatorA = new SimpleDescriptorEvaluator();
            var evaluatorB = new SimpleDescriptorEvaluator();

            using var mapA = new MapElites<DummyContext>(
                parameterCount: 8,
                batchSize: 64,
                archive: archiveA,
                evaluator: evaluatorA,
                seed: 777,
                mutationSigma: 0.05f,
                initialMin: -1f,
                initialMax: 1f,
                randomInjectionProbability: 0.15f);

            using var mapB = new MapElites<DummyContext>(
                parameterCount: 8,
                batchSize: 64,
                archive: archiveB,
                evaluator: evaluatorB,
                seed: 777,
                mutationSigma: 0.05f,
                initialMin: -1f,
                initialMax: 1f,
                randomInjectionProbability: 0.15f);

            var ctxA = new DummyContext();
            var ctxB = new DummyContext();

            MapElitesIterationMetrics lastA = default;
            MapElitesIterationMetrics lastB = default;

            for (var i = 0; i < 20; i++)
            {
                lastA = mapA.RunIteration(ref ctxA);
                lastB = mapB.RunIteration(ref ctxB);

                Assert.Equal(lastA.Iteration, lastB.Iteration);
                Assert.Equal(lastA.InsertedNewCells, lastB.InsertedNewCells);
                Assert.Equal(lastA.ReplacedExistingCells, lastB.ReplacedExistingCells);
                Assert.Equal(lastA.RejectedCount, lastB.RejectedCount);
                Assert.Equal(lastA.OutOfBoundsCount, lastB.OutOfBoundsCount);
                Assert.Equal(lastA.OccupiedCells, lastB.OccupiedCells);
                Assert.Equal(lastA.Coverage, lastB.Coverage, 6);
                Assert.Equal(lastA.QdScore, lastB.QdScore, 5);
                Assert.Equal(lastA.BestEvaluatedFitness, lastB.BestEvaluatedFitness, 5);
                Assert.Equal(lastA.BestEliteFitness, lastB.BestEliteFitness, 5);
            }

            Assert.Equal(mapA.GetBestEvaluatedParameters().ToArray(), mapB.GetBestEvaluatedParameters().ToArray());
        }

        [Fact]
        public void Coverage_GrowsOverMultipleIterations_OnSimpleProblem()
        {
            using var archive = CreateArchive(parameterCount: 8);
            var evaluator = new SimpleDescriptorEvaluator();

            using var map = new MapElites<DummyContext>(
                parameterCount: 8,
                batchSize: 128,
                archive: archive,
                evaluator: evaluator,
                seed: 42,
                mutationSigma: 0.08f,
                initialMin: -1f,
                initialMax: 1f,
                randomInjectionProbability: 0.10f);

            var ctx = new DummyContext();

            var initialCoverage = 0f;
            var finalCoverage = 0f;

            for (var i = 0; i < 40; i++)
            {
                var metrics = map.RunIteration(ref ctx);
                if (i == 0)
                {
                    initialCoverage = metrics.Coverage;
                }

                finalCoverage = metrics.Coverage;
            }

            Assert.True(finalCoverage >= initialCoverage);
            Assert.True(finalCoverage > 0.10f, $"Expected coverage > 0.10, got {finalCoverage:F4}.");
            Assert.True(archive.OccupiedCount > 0);
        }

        [Fact]
        public void BetterCandidate_ReplacesEliteInSameCell()
        {
            using var archive = new GridEliteArchive(
                parameterCount: 4,
                binsPerDimension: stackalloc[] { 4, 4 },
                descriptorMin: stackalloc[] { 0f, 0f },
                descriptorMax: stackalloc[] { 1f, 1f });

            var evaluator = new FixedCellEvaluator();
            using var map = new MapElites<DummyContext>(
                parameterCount: 4,
                batchSize: 2,
                archive: archive,
                evaluator: evaluator,
                seed: 1,
                mutationSigma: 0.01f,
                initialMin: -0.1f,
                initialMax: 0.1f,
                randomInjectionProbability: 0f);

            float[] population =
            [
                1f, 1f, 1f, 1f,
                2f, 2f, 2f, 2f
            ];

            float[] fitness = [1f, 5f];

            float[] descriptors =
            [
                0.30f, 0.30f,
                0.30f, 0.30f
            ];

            var metrics = map.Tell(
                populationMatrix: population,
                fitness: fitness,
                descriptors: descriptors);

            Assert.Equal(1, metrics.InsertedNewCells);
            Assert.Equal(1, metrics.ReplacedExistingCells);
            Assert.Equal(0, metrics.RejectedCount);

            Assert.True(archive.TryGetCellIndex([0.30f, 0.30f], out var cellIndex));
            Assert.Equal(5f, archive.GetFitness(cellIndex), 6);
            Assert.Equal(new[] { 2f, 2f, 2f, 2f }, archive.GetParameters(cellIndex).ToArray());
        }

        private static GridEliteArchive CreateArchive(int parameterCount)
        {
            return new GridEliteArchive(
                parameterCount: parameterCount,
                binsPerDimension: stackalloc[] { 16, 16 },
                descriptorMin: stackalloc[] { 0f, 0f },
                descriptorMax: stackalloc[] { 1f, 1f });
        }

        private readonly struct DummyContext
        {
        }

        private sealed class SimpleDescriptorEvaluator : IBehaviorDescriptorEvaluator<DummyContext>
        {
            public float Evaluate(
                ReadOnlySpan<float> parameters,
                ref DummyContext context,
                Span<float> descriptor)
            {
                if (descriptor.Length != 2)
                {
                    throw new ArgumentException("Descriptor length must be 2.", nameof(descriptor));
                }

                var sumSq = 0f;
                for (var i = 0; i < parameters.Length; i++)
                {
                    sumSq += parameters[i] * parameters[i];
                }

                descriptor[0] = Clamp01(0.5f + (0.25f * parameters[0]));
                descriptor[1] = Clamp01(0.5f + (0.25f * parameters[1]));

                return -sumSq;
            }

            private static float Clamp01(float value)
            {
                if (value < 0f) return 0f;
                if (value > 1f) return 1f;
                return value;
            }
        }

        private sealed class FixedCellEvaluator : IBehaviorDescriptorEvaluator<DummyContext>
        {
            public float Evaluate(
                ReadOnlySpan<float> parameters,
                ref DummyContext context,
                Span<float> descriptor)
            {
                descriptor[0] = 0.30f;
                descriptor[1] = 0.30f;

                var fitness = 0f;
                for (var i = 0; i < parameters.Length; i++)
                {
                    fitness += parameters[i];
                }

                return fitness;
            }
        }
    }
}