// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Evolutionary.Storage
{
    /// <summary>
    /// Shared pooled workspace for evolutionary algorithms.
    /// Keeps all hot-path buffers in one place and exposes convenient span accessors.
    ///
    /// Backward-compatible with the current GA usage:
    /// - Population
    /// - NextPopulation
    /// - Fitness
    /// - ShapedFitness
    /// - Ranking
    /// - EliteIndices
    /// </summary>
    public sealed class EvolutionWorkspace : IDisposable
    {
        private int _disposed;
        private FastTensor<float> _population;
        private FastTensor<float> _nextPopulation;

        public EvolutionWorkspace(int populationSize, int genomeSize, bool clearMemory = false)
        {
            if (populationSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(populationSize));
            }

            if (genomeSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(genomeSize));
            }

            PopulationSize = populationSize;
            GenomeSize = genomeSize;

            _population = new FastTensor<float>(populationSize, genomeSize, clearMemory);
            _nextPopulation = new FastTensor<float>(populationSize, genomeSize, clearMemory);
            Fitness = new FastTensor<float>(populationSize, clearMemory);
            ShapedFitness = new FastTensor<float>(populationSize, clearMemory);

            Ranking = new int[populationSize];
            EliteIndices = new int[populationSize];

            ResetIndices();
        }

        public int PopulationSize { get; }

        public int GenomeSize { get; }

        public int PopulationMatrixLength => PopulationSize * GenomeSize;

        // Backward-compatible object access
        public FastTensor<float> Population
        {
            get
            {
                ThrowIfDisposed();
                return _population;
            }
        }

        public FastTensor<float> NextPopulation
        {
            get
            {
                ThrowIfDisposed();
                return _nextPopulation;
            }
        }

        public FastTensor<float> Fitness
        {
            get
            {
                ThrowIfDisposed();
                return _fitnessTensor;
            }
            private init => _fitnessTensor = value;
        }

        public FastTensor<float> ShapedFitness
        {
            get
            {
                ThrowIfDisposed();
                return _shapedFitnessTensor;
            }
            private init => _shapedFitnessTensor = value;
        }

        public int[] Ranking { get; }

        public int[] EliteIndices { get; }

        private readonly FastTensor<float> _fitnessTensor = null!;
        private readonly FastTensor<float> _shapedFitnessTensor = null!;

        // Fast span accessors for runners / evaluators / algorithms
        public Span<float> PopulationSpan
        {
            get
            {
                ThrowIfDisposed();
                return _population.GetView().AsSpan();
            }
        }

        public Span<float> NextPopulationSpan
        {
            get
            {
                ThrowIfDisposed();
                return _nextPopulation.GetView().AsSpan();
            }
        }

        public Span<float> FitnessSpan
        {
            get
            {
                ThrowIfDisposed();
                return _fitnessTensor.GetView().AsSpan();
            }
        }

        public Span<float> ShapedFitnessSpan
        {
            get
            {
                ThrowIfDisposed();
                return _shapedFitnessTensor.GetView().AsSpan();
            }
        }

        public ReadOnlySpan<float> PopulationReadOnlySpan
        {
            get
            {
                ThrowIfDisposed();
                return _population.GetView().AsReadOnlySpan();
            }
        }

        public ReadOnlySpan<float> NextPopulationReadOnlySpan
        {
            get
            {
                ThrowIfDisposed();
                return _nextPopulation.GetView().AsReadOnlySpan();
            }
        }

        public ReadOnlySpan<float> FitnessReadOnlySpan
        {
            get
            {
                ThrowIfDisposed();
                return _fitnessTensor.GetView().AsReadOnlySpan();
            }
        }

        public ReadOnlySpan<float> ShapedFitnessReadOnlySpan
        {
            get
            {
                ThrowIfDisposed();
                return _shapedFitnessTensor.GetView().AsReadOnlySpan();
            }
        }

        public void SwapPopulations()
        {
            ThrowIfDisposed();
            (_population, _nextPopulation) = (_nextPopulation, _population);
        }

        public void ClearPopulation()
        {
            ThrowIfDisposed();
            PopulationSpan.Clear();
        }

        public void ClearNextPopulation()
        {
            ThrowIfDisposed();
            NextPopulationSpan.Clear();
        }

        public void ClearFitness()
        {
            ThrowIfDisposed();
            FitnessSpan.Clear();
        }

        public void ClearShapedFitness()
        {
            ThrowIfDisposed();
            ShapedFitnessSpan.Clear();
        }

        public void ClearAll()
        {
            ThrowIfDisposed();
            PopulationSpan.Clear();
            NextPopulationSpan.Clear();
            FitnessSpan.Clear();
            ShapedFitnessSpan.Clear();
            ResetIndices();
        }

        public void ResetIndices()
        {
            ThrowIfDisposed();

            for (var i = 0; i < PopulationSize; i++)
            {
                Ranking[i] = i;
                EliteIndices[i] = i;
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            _population.Dispose();
            _nextPopulation.Dispose();
            _fitnessTensor.Dispose();
            _shapedFitnessTensor.Dispose();
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);
        }
    }
}