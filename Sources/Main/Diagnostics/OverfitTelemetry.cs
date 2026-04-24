// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Diagnostics.Contracts;

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

        // Existing DL/runtime metrics
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

        public static readonly Counter<long> GraphRecordTotalCount = Meter.CreateCounter<long>(
            "overfit.graph.record_op.count",
            unit: "{op}",
            description: "Graph record record total count");

        // Evolutionary runner metrics
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Activity StartActivity(string name, ActivityKind kind = ActivityKind.Internal)
        {
            if (!Enabled || !TraceEnabled)
            {
                return null;
            }

            return Tracer.StartActivity(name, kind);
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
        public static void RecordGraphRecordOp(
            OpCode code)
        {
            if (!Enabled)
            {
                return;
            }

            TagList tags = default;
            tags.Add("op", code);

            GraphRecordTotalCount.Add(1, tags);
        }
    }
}