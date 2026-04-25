// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.IO;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Diagnostics.Contracts;
using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Storage;

namespace DevOnBike.Overfit.Evolutionary.Runtime
{
    /// <summary>
    /// Minimal orchestration loop for evolutionary algorithms using Ask/Evaluate/Tell.
    /// Keeps one reusable workspace and reports generation metrics to OverfitTelemetry.
    /// </summary>
    public sealed class EvolutionRunner : IDisposable
    {
        private readonly IEvolutionAlgorithm _algorithm;
        private readonly IPopulationEvaluator _evaluator;
        private readonly EvolutionWorkspace _workspace;
        private readonly bool _ownsWorkspace;
        private bool _disposed;

        public EvolutionRunner(
            IEvolutionAlgorithm algorithm,
            IPopulationEvaluator evaluator,
            EvolutionWorkspace? workspace = null)
        {
            _algorithm = algorithm ?? throw new ArgumentNullException(nameof(algorithm));
            _evaluator = evaluator ?? throw new ArgumentNullException(nameof(evaluator));

            if (workspace is null)
            {
                _workspace = new EvolutionWorkspace(
                    _algorithm.PopulationSize,
                    _algorithm.ParameterCount,
                    clearMemory: false);

                _ownsWorkspace = true;
            }
            else
            {
                if (workspace.PopulationSize != _algorithm.PopulationSize)
                {
                    throw new ArgumentException(
                        $"Workspace population size {workspace.PopulationSize} does not match algorithm population size {_algorithm.PopulationSize}.",
                        nameof(workspace));
                }

                if (workspace.GenomeSize != _algorithm.ParameterCount)
                {
                    throw new ArgumentException(
                        $"Workspace genome size {workspace.GenomeSize} does not match algorithm parameter count {_algorithm.ParameterCount}.",
                        nameof(workspace));
                }

                _workspace = workspace;
                _ownsWorkspace = false;
            }
        }

        public IEvolutionAlgorithm Algorithm
        {
            get
            {
                ThrowIfDisposed();
                return _algorithm;
            }
        }

        public IPopulationEvaluator Evaluator
        {
            get
            {
                ThrowIfDisposed();
                return _evaluator;
            }
        }

        public EvolutionWorkspace Workspace
        {
            get
            {
                ThrowIfDisposed();
                return _workspace;
            }
        }

        public EvolutionGenerationMetrics RunGeneration()
        {
            ThrowIfDisposed();

            using var activity = OverfitTelemetry.StartActivity("evolution.generation");

            var totalSw = ValueStopwatch.StartNew();

            var askSw = ValueStopwatch.StartNew();
            _algorithm.Ask(_workspace.PopulationSpan);
            var askElapsed = askSw.GetElapsedTime();

            var evaluateSw = ValueStopwatch.StartNew();
            _evaluator.Evaluate(
                _workspace.PopulationReadOnlySpan,
                _workspace.FitnessSpan,
                _algorithm.PopulationSize,
                _algorithm.ParameterCount);
            var evaluateElapsed = evaluateSw.GetElapsedTime();

            var tellSw = ValueStopwatch.StartNew();
            _algorithm.Tell(_workspace.FitnessReadOnlySpan);
            var tellElapsed = tellSw.GetElapsedTime();

            var totalElapsed = totalSw.GetElapsedTime();

            var metrics = new EvolutionGenerationMetrics(
                generation: _algorithm.Generation,
                bestFitness: _algorithm.BestFitness,
                totalElapsed: totalElapsed,
                askElapsed: askElapsed,
                evaluateElapsed: evaluateElapsed,
                tellElapsed: tellElapsed);

            OverfitTelemetry.RecordEvolutionGeneration(
                metrics,
                _algorithm.PopulationSize,
                _algorithm.ParameterCount);

            if (activity is not null)
            {
                activity.SetTag("generation", metrics.Generation);
                activity.SetTag("population_size", _algorithm.PopulationSize);
                activity.SetTag("parameter_count", _algorithm.ParameterCount);
                activity.SetTag("best_fitness", metrics.BestFitness);
                activity.SetTag("duration_ms", metrics.TotalElapsed.TotalMilliseconds);
                activity.SetTag("ask_ms", metrics.AskElapsed.TotalMilliseconds);
                activity.SetTag("evaluate_ms", metrics.EvaluateElapsed.TotalMilliseconds);
                activity.SetTag("tell_ms", metrics.TellElapsed.TotalMilliseconds);
            }

            return metrics;
        }

        public void Run(
            int generations,
            Action<EvolutionGenerationMetrics>? onGenerationCompleted = null,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            if (generations < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(generations));
            }

            for (var i = 0; i < generations; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var metrics = RunGeneration();
                onGenerationCompleted?.Invoke(metrics);
            }
        }

        public void Run(
            int generations,
            int checkpointEvery,
            Func<int, string?> checkpointPathFactory,
            Action<EvolutionGenerationMetrics>? onGenerationCompleted = null,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            if (generations < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(generations));
            }

            if (checkpointEvery <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(checkpointEvery));
            }

            ArgumentNullException.ThrowIfNull(checkpointPathFactory);

            for (var i = 0; i < generations; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var metrics = RunGeneration();
                onGenerationCompleted?.Invoke(metrics);

                if (metrics.Generation > 0 && metrics.Generation % checkpointEvery == 0)
                {
                    var path = checkpointPathFactory(metrics.Generation);
                    if (!string.IsNullOrWhiteSpace(path))
                    {
                        SaveCheckpoint(path!);
                    }
                }
            }
        }

        public void SaveCheckpoint(string path)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(path);

            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            using var stream = File.Create(path);
            using var writer = new BinaryWriter(stream);
            _algorithm.Save(writer);
        }

        public void LoadCheckpoint(string path)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(path);

            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);
            _algorithm.Load(reader);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            if (_ownsWorkspace)
            {
                _workspace.Dispose();
            }

            _algorithm.Dispose();
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}