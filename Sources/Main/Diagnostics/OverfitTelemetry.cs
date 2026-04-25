// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Diagnostics.Contracts;
using DevOnBike.Overfit.Evolutionary.Runtime;

namespace DevOnBike.Overfit.Diagnostics
{
    public static class OverfitTelemetry
    {
        public const string MeterName = "DevOnBike.Overfit";
        public const string Version = "1.0.0";

        public static bool Enabled { get; set; } = true;
        public static bool TraceEnabled { get; set; } = false;

        public static readonly Meter Meter = new(MeterName, Version);
        public static readonly ActivitySource Tracer = new(MeterName, Version);

        // Existing runtime metrics
        public static readonly Histogram<double> KernelDurationMs = Meter.CreateHistogram<double>(
            "overfit.kernel.duration.ms",
            unit: "ms",
            description: "Execution time of low-level kernels.");

        public static readonly Histogram<double> ModuleDurationMs = Meter.CreateHistogram<double>(
            "overfit.module.duration.ms",
            unit: "ms",
            description: "Execution time of high-level DL modules.");

        public static readonly Histogram<double> GraphBackwardDurationMs = Meter.CreateHistogram<double>(
            "overfit.graph.backward.duration.ms",
            unit: "ms",
            description: "Backward pass duration.");

        public static readonly Counter<long> GraphRecordTotalCount = Meter.CreateCounter<long>(
            "overfit.graph.record_op.count",
            unit: "{op}",
            description: "Graph record op total count.");

        public static readonly Histogram<long> ModuleAllocatedBytes = Meter.CreateHistogram<long>(
            "overfit.module.alloc.bytes",
            unit: "By",
            description: "Allocated managed bytes captured for module execution.");

        public static readonly Histogram<long> GraphAllocatedBytes = Meter.CreateHistogram<long>(
            "overfit.graph.alloc.bytes",
            unit: "By",
            description: "Allocated managed bytes captured during backward.");

        public static readonly Histogram<long> AllocationBytes = Meter.CreateHistogram<long>(
            "overfit.allocation.bytes",
            unit: "By",
            description: "Reported tensor/buffer allocations.");

        public static readonly Counter<long> KernelCount = Meter.CreateCounter<long>(
            "overfit.kernel.count",
            unit: "{kernel}",
            description: "Number of kernel executions.");

        public static readonly Counter<long> ModuleCount = Meter.CreateCounter<long>(
            "overfit.module.count",
            unit: "{module}",
            description: "Number of module executions.");

        public static readonly Counter<long> GraphCount = Meter.CreateCounter<long>(
            "overfit.graph.count",
            unit: "{graph}",
            description: "Number of graph completion events.");

        public static readonly Counter<long> TapeOpCount = Meter.CreateCounter<long>(
            "overfit.graph.tape_ops",
            unit: "{op}",
            description: "Total number of tape ops recorded in completed graphs.");

        public static readonly UpDownCounter<long> NativeMemoryBytes = Meter.CreateUpDownCounter<long>(
            "overfit.memory.native.bytes",
            unit: "By",
            description: "Native / unmanaged memory tracked by Overfit diagnostics.");

        // --------------------------------------------------------------------
        // Low-level tensor storage infrastructure metrics.
        //
        // These are intentionally context-free. TensorStorage knows only that
        // physical storage was created/disposed. It does not know whether the
        // storage is a graph intermediate, gradient, optimizer buffer, scratch,
        // user tensor, or evolutionary workspace.
        // --------------------------------------------------------------------

        public static readonly Counter<long> TensorStorageCreated = Meter.CreateCounter<long>(
            "overfit.tensor_storage.created",
            unit: "{storage}",
            description: "Total number of TensorStorage instances created.");

        public static readonly Counter<long> TensorStorageDisposed = Meter.CreateCounter<long>(
            "overfit.tensor_storage.disposed",
            unit: "{storage}",
            description: "Total number of TensorStorage instances disposed.");

        public static readonly Counter<long> TensorStoragePooledCreated = Meter.CreateCounter<long>(
            "overfit.tensor_storage.pooled.created",
            unit: "{storage}",
            description: "Number of pooled TensorStorage instances created.");

        public static readonly Counter<long> TensorStorageBorrowedCreated = Meter.CreateCounter<long>(
            "overfit.tensor_storage.borrowed.created",
            unit: "{storage}",
            description: "Number of borrowed arena TensorStorage instances created.");

        public static readonly Counter<long> TensorStoragePooledDisposed = Meter.CreateCounter<long>(
            "overfit.tensor_storage.pooled.disposed",
            unit: "{storage}",
            description: "Number of pooled TensorStorage instances disposed.");

        public static readonly Counter<long> TensorStorageBorrowedDisposed = Meter.CreateCounter<long>(
            "overfit.tensor_storage.borrowed.disposed",
            unit: "{storage}",
            description: "Number of borrowed arena TensorStorage instances disposed.");

        public static readonly Counter<long> TensorStorageElementsCreated = Meter.CreateCounter<long>(
            "overfit.tensor_storage.elements.created",
            unit: "{element}",
            description: "Total number of elements requested by created TensorStorage instances.");

        public static readonly Counter<long> TensorStorageBytesCreated = Meter.CreateCounter<long>(
            "overfit.tensor_storage.bytes.created",
            unit: "By",
            description: "Estimated bytes requested by created TensorStorage instances.");

        // Evolutionary generation metrics
        public static readonly Histogram<double> EvolutionGenerationDurationMs = Meter.CreateHistogram<double>(
            "overfit.evolution.generation.duration.ms",
            unit: "ms",
            description: "Total duration of one evolutionary generation.");

        public static readonly Histogram<double> EvolutionAskDurationMs = Meter.CreateHistogram<double>(
            "overfit.evolution.ask.duration.ms",
            unit: "ms",
            description: "Duration of the Ask phase.");

        public static readonly Histogram<double> EvolutionEvaluateDurationMs = Meter.CreateHistogram<double>(
            "overfit.evolution.evaluate.duration.ms",
            unit: "ms",
            description: "Duration of the Evaluate phase.");

        public static readonly Histogram<double> EvolutionTellDurationMs = Meter.CreateHistogram<double>(
            "overfit.evolution.tell.duration.ms",
            unit: "ms",
            description: "Duration of the Tell phase.");

        public static readonly Histogram<double> EvolutionBestFitness = Meter.CreateHistogram<double>(
            "overfit.evolution.best_fitness",
            unit: "{fitness}",
            description: "Best fitness observed after a generation.");

        public static readonly Counter<long> EvolutionGenerationCount = Meter.CreateCounter<long>(
            "overfit.evolution.generation.count",
            unit: "{generation}",
            description: "Number of completed evolutionary generations.");

        public static readonly Counter<long> EvolutionPopulationEvaluated = Meter.CreateCounter<long>(
            "overfit.evolution.population.evaluated",
            unit: "{candidate}",
            description: "Number of candidates evaluated by the runner.");

        // MAP-Elites metrics
        public static readonly Histogram<double> MapElitesIterationDurationMs = Meter.CreateHistogram<double>(
            "overfit.evolution.map_elites.iteration.duration.ms",
            unit: "ms",
            description: "Duration of one MAP-Elites iteration.");

        public static readonly Histogram<double> MapElitesAskDurationMs = Meter.CreateHistogram<double>(
            "overfit.evolution.map_elites.ask.duration.ms",
            unit: "ms",
            description: "Duration of MAP-Elites Ask phase.");

        public static readonly Histogram<double> MapElitesEvaluateDurationMs = Meter.CreateHistogram<double>(
            "overfit.evolution.map_elites.evaluate.duration.ms",
            unit: "ms",
            description: "Duration of MAP-Elites Evaluate phase.");

        public static readonly Histogram<double> MapElitesTellDurationMs = Meter.CreateHistogram<double>(
            "overfit.evolution.map_elites.tell.duration.ms",
            unit: "ms",
            description: "Duration of MAP-Elites Tell phase.");

        public static readonly Histogram<double> MapElitesCoverage = Meter.CreateHistogram<double>(
            "overfit.evolution.map_elites.coverage",
            unit: "{coverage}",
            description: "Archive coverage after MAP-Elites iteration.");

        public static readonly Histogram<double> MapElitesQdScore = Meter.CreateHistogram<double>(
            "overfit.evolution.map_elites.qd_score",
            unit: "{score}",
            description: "QD score after MAP-Elites iteration.");

        public static readonly Histogram<double> MapElitesBestFitness = Meter.CreateHistogram<double>(
            "overfit.evolution.map_elites.best_fitness",
            unit: "{fitness}",
            description: "Best fitness after MAP-Elites iteration.");

        public static readonly Counter<long> MapElitesIterationCount = Meter.CreateCounter<long>(
            "overfit.evolution.map_elites.iteration.count",
            unit: "{iteration}",
            description: "Number of completed MAP-Elites iterations.");

        public static readonly Counter<long> MapElitesInsertedNewCells = Meter.CreateCounter<long>(
            "overfit.evolution.map_elites.inserted_new_cells",
            unit: "{cell}",
            description: "Number of newly occupied archive cells.");

        public static readonly Counter<long> MapElitesReplacedCells = Meter.CreateCounter<long>(
            "overfit.evolution.map_elites.replaced_cells",
            unit: "{cell}",
            description: "Number of elite replacements.");

        public static readonly Counter<long> MapElitesRejectedCandidates = Meter.CreateCounter<long>(
            "overfit.evolution.map_elites.rejected_candidates",
            unit: "{candidate}",
            description: "Number of rejected MAP-Elites candidates.");

        public static readonly Counter<long> MapElitesOutOfBoundsCandidates = Meter.CreateCounter<long>(
            "overfit.evolution.map_elites.out_of_bounds_candidates",
            unit: "{candidate}",
            description: "Number of MAP-Elites candidates whose descriptors were outside archive bounds.");

        public static readonly Histogram<long> MapElitesOccupiedCells = Meter.CreateHistogram<long>(
            "overfit.evolution.map_elites.occupied_cells",
            unit: "{cell}",
            description: "Number of occupied archive cells.");

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Activity? StartActivity(string name, ActivityKind kind = ActivityKind.Internal)
        {
            if (!Enabled || !TraceEnabled)
            {
                return null;
            }

            return Tracer.StartActivity(name, kind);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RecordGraphRecordOp(OpCode code)
        {
            if (!Enabled)
            {
                return;
            }

            TagList tags = default;
            tags.Add("op", code.ToString());

            GraphRecordTotalCount.Add(1, tags);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RecordTensorStorageCreated(
            int length,
            int elementSizeBytes,
            bool borrowed)
        {
            if (!Enabled)
            {
                return;
            }

            TensorStorageCreated.Add(1);
            TensorStorageElementsCreated.Add(length);
            TensorStorageBytesCreated.Add((long)length * elementSizeBytes);

            if (borrowed)
            {
                TensorStorageBorrowedCreated.Add(1);
            }
            else
            {
                TensorStoragePooledCreated.Add(1);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RecordTensorStorageDisposed(bool borrowed)
        {
            if (!Enabled)
            {
                return;
            }

            TensorStorageDisposed.Add(1);

            if (borrowed)
            {
                TensorStorageBorrowedDisposed.Add(1);
            }
            else
            {
                TensorStoragePooledDisposed.Add(1);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RecordEvolutionGeneration(
            in EvolutionGenerationMetrics metrics,
            int populationSize,
            int parameterCount)
        {
            if (!Enabled)
            {
                return;
            }

            TagList tags = default;
            tags.Add("population_size", populationSize);
            tags.Add("parameter_count", parameterCount);

            EvolutionGenerationDurationMs.Record(metrics.TotalElapsed.TotalMilliseconds, tags);
            EvolutionAskDurationMs.Record(metrics.AskElapsed.TotalMilliseconds, tags);
            EvolutionEvaluateDurationMs.Record(metrics.EvaluateElapsed.TotalMilliseconds, tags);
            EvolutionTellDurationMs.Record(metrics.TellElapsed.TotalMilliseconds, tags);
            EvolutionBestFitness.Record(metrics.BestFitness, tags);
            EvolutionGenerationCount.Add(1, tags);
            EvolutionPopulationEvaluated.Add(populationSize, tags);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RecordMapElitesIteration(
            in MapElitesIterationMetrics metrics,
            int batchSize,
            int parameterCount,
            int descriptorDimensions)
        {
            if (!Enabled)
            {
                return;
            }

            TagList tags = default;
            tags.Add("batch_size", batchSize);
            tags.Add("parameter_count", parameterCount);
            tags.Add("descriptor_dimensions", descriptorDimensions);

            MapElitesIterationDurationMs.Record(metrics.TotalElapsed.TotalMilliseconds, tags);
            MapElitesAskDurationMs.Record(metrics.AskElapsed.TotalMilliseconds, tags);
            MapElitesEvaluateDurationMs.Record(metrics.EvaluateElapsed.TotalMilliseconds, tags);
            MapElitesTellDurationMs.Record(metrics.TellElapsed.TotalMilliseconds, tags);

            MapElitesCoverage.Record(metrics.Coverage, tags);
            MapElitesQdScore.Record(metrics.QdScore, tags);
            MapElitesBestFitness.Record(metrics.BestFitness, tags);

            MapElitesIterationCount.Add(1, tags);
            MapElitesInsertedNewCells.Add(metrics.InsertedNewCells, tags);
            MapElitesReplacedCells.Add(metrics.ReplacedExistingCells, tags);
            MapElitesRejectedCandidates.Add(metrics.RejectedCount, tags);
            MapElitesOutOfBoundsCandidates.Add(metrics.OutOfBoundsCount, tags);
            MapElitesOccupiedCells.Record(metrics.OccupiedCells, tags);
        }
    }
}