// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Fitness;
using DevOnBike.Overfit.Evolutionary.Mutation;
using DevOnBike.Overfit.Evolutionary.Selection;
using DevOnBike.Overfit.Evolutionary.Storage;
using DevOnBike.Overfit.Evolutionary.Strategies;

namespace DevOnBike.Overfit.Tests
{
    public sealed class EvolutionaryCheckpointTests
    {
        // ----------------------------------------------------------------------
        // GA
        // ----------------------------------------------------------------------

        [Fact]
        public void Ga_Checkpoint_Roundtrip_RestoresObservableState()
        {
            using var original = BuildGa(seed: 1);
            original.Initialize();

            var pop = new float[original.PopulationSize * original.ParameterCount];
            var fitness = new float[original.PopulationSize];
            var rng = new Random(42);

            for (var gen = 0; gen < 5; gen++)
            {
                original.Ask(pop);
                for (var i = 0; i < fitness.Length; i++)
                {
                    fitness[i] = rng.NextSingle();
                }
                original.Tell(fitness);
            }

            var savedPopulation = new float[pop.Length];
            original.Ask(savedPopulation);
            var savedBest = original.GetBestParameters().ToArray();
            var savedBestFitness = original.BestFitness;
            var savedGeneration = original.Generation;

            byte[] checkpointBytes;
            using (var ms = new MemoryStream())
            {
                using var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true);
                original.Save(bw);
                checkpointBytes = ms.ToArray();
            }

            using var restored = BuildGa(seed: 999); // different seed on purpose
            restored.Initialize();                   // populate with different random data

            using (var ms = new MemoryStream(checkpointBytes))
            {
                using var br = new BinaryReader(ms);
                restored.Load(br);
            }

            Assert.Equal(savedGeneration, restored.Generation);
            Assert.Equal(savedBestFitness, restored.BestFitness, 5);

            var restoredBest = restored.GetBestParameters().ToArray();
            Assert.Equal(savedBest, restoredBest);

            var restoredPopulation = new float[pop.Length];
            restored.Ask(restoredPopulation);
            Assert.Equal(savedPopulation, restoredPopulation);
        }

        [Fact]
        public void Ga_Load_RejectsWrongMagic()
        {
            using var ga = BuildGa(seed: 1);
            ga.Initialize();

            var garbage = new byte[256];
            garbage[0] = 0x42; // clearly not our magic

            using var ms = new MemoryStream(garbage);
            using var br = new BinaryReader(ms);

            Assert.Throws<InvalidDataException>(() => ga.Load(br));
        }

        [Fact]
        public void Ga_Load_RejectsMismatchedDimensions()
        {
            using var source = new GenerationalGeneticAlgorithm(
                populationSize: 16,
                parameterCount: 4,
                eliteFraction: 0.25f,
                selectionOperator: new TruncationSelectionOperator(),
                mutationOperator: new GaussianMutationOperator(),
                seed: 1);
            source.Initialize();

            byte[] bytes;
            using (var ms = new MemoryStream())
            {
                using var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true);
                source.Save(bw);
                bytes = ms.ToArray();
            }

            using var target = new GenerationalGeneticAlgorithm(
                populationSize: 32,  // different
                parameterCount: 4,
                eliteFraction: 0.25f,
                selectionOperator: new TruncationSelectionOperator(),
                mutationOperator: new GaussianMutationOperator(),
                seed: 2);

            using var msLoad = new MemoryStream(bytes);
            using var br = new BinaryReader(msLoad);

            Assert.Throws<InvalidDataException>(() => target.Load(br));
        }

        [Fact]
        public void Ga_Checkpoint_WithoutFitness_RoundTripsCleanly()
        {
            // Saving immediately after Initialize (no Tell yet) must produce a valid
            // stream that Load can consume without tripping over the missing fitness data.
            using var original = BuildGa(seed: 1);
            original.Initialize();

            byte[] bytes;
            using (var ms = new MemoryStream())
            {
                using var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true);
                original.Save(bw);
                bytes = ms.ToArray();
            }

            using var restored = BuildGa(seed: 2);
            restored.Initialize();

            using var msLoad = new MemoryStream(bytes);
            using var br = new BinaryReader(msLoad);
            restored.Load(br);

            Assert.Equal(0, restored.Generation);
            Assert.True(float.IsNaN(restored.BestFitness));
            Assert.True(restored.GetBestParameters().IsEmpty);
        }

        // ----------------------------------------------------------------------
        // ES
        // ----------------------------------------------------------------------

        [Fact]
        public void Es_Checkpoint_Roundtrip_RestoresMuAndBestState()
        {
            var noise = new PrecomputedNoiseTable(length: 1 << 14, seed: 11);

            using var original = new OpenAiEsStrategy(
                populationSize: 16,
                parameterCount: 8,
                sigma: 0.1f,
                learningRate: 0.02f,
                noiseTable: noise,
                seed: 1);
            original.Initialize();

            var pop = new float[16 * 8];
            var fitness = new float[16];
            var rng = new Random(42);

            for (var gen = 0; gen < 5; gen++)
            {
                original.Ask(pop);
                for (var i = 0; i < fitness.Length; i++)
                {
                    fitness[i] = rng.NextSingle();
                }
                original.Tell(fitness);
            }

            var savedMu = original.Mean.ToArray();
            var savedBest = original.GetBestParameters().ToArray();
            var savedBestFitness = original.BestFitness;
            var savedGeneration = original.Generation;

            byte[] bytes;
            using (var ms = new MemoryStream())
            {
                using var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true);
                original.Save(bw);
                bytes = ms.ToArray();
            }

            using var restored = new OpenAiEsStrategy(
                populationSize: 16,
                parameterCount: 8,
                sigma: 0.1f,
                learningRate: 0.02f,
                noiseTable: noise,
                seed: 999);
            restored.Initialize();

            using var msLoad = new MemoryStream(bytes);
            using var br = new BinaryReader(msLoad);
            restored.Load(br);

            Assert.Equal(savedGeneration, restored.Generation);
            Assert.Equal(savedBestFitness, restored.BestFitness, 5);
            Assert.Equal(savedMu, restored.Mean.ToArray());
            Assert.Equal(savedBest, restored.GetBestParameters().ToArray());
        }

        [Fact]
        public void Es_Load_RejectsMismatchedAdamMode()
        {
            // A checkpoint written by Adam mode must not load into an SGD-mode instance
            // (and vice versa): the downstream update rule would diverge silently.
            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);

            using var adamEs = new OpenAiEsStrategy(
                populationSize: 8, parameterCount: 4,
                sigma: 0.1f, learningRate: 0.01f,
                noiseTable: noise, useAdam: true);
            adamEs.Initialize();

            byte[] adamBytes;
            using (var ms = new MemoryStream())
            {
                using var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true);
                adamEs.Save(bw);
                adamBytes = ms.ToArray();
            }

            using var sgdEs = new OpenAiEsStrategy(
                populationSize: 8, parameterCount: 4,
                sigma: 0.1f, learningRate: 0.01f,
                noiseTable: noise, useAdam: false);

            using var msLoad = new MemoryStream(adamBytes);
            using var br = new BinaryReader(msLoad);

            Assert.Throws<InvalidDataException>(() => sgdEs.Load(br));
        }

        [Fact]
        public void Es_Load_RejectsGaCheckpoint()
        {
            // A checkpoint produced by the GA must not load into an ES instance, because the
            // magic differs. Protects users from silently using the wrong algorithm after
            // pointing an ES process at a file written by a GA process.
            using var ga = BuildGa(seed: 1);
            ga.Initialize();

            byte[] gaBytes;
            using (var ms = new MemoryStream())
            {
                using var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true);
                ga.Save(bw);
                gaBytes = ms.ToArray();
            }

            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);
            using var es = new OpenAiEsStrategy(
                populationSize: 8,
                parameterCount: 4,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: noise);

            using var msLoad = new MemoryStream(gaBytes);
            using var br = new BinaryReader(msLoad);

            Assert.Throws<InvalidDataException>(() => es.Load(br));
        }

        // ----------------------------------------------------------------------
        // Helpers
        // ----------------------------------------------------------------------

        private static GenerationalGeneticAlgorithm BuildGa(int seed)
        {
            return new GenerationalGeneticAlgorithm(
                populationSize: 16,
                parameterCount: 4,
                eliteFraction: 0.25f,
                selectionOperator: new TruncationSelectionOperator(),
                mutationOperator: new GaussianMutationOperator(),
                fitnessShaper: new CenteredRankFitnessShaper(),
                seed: seed);
        }
    }
}