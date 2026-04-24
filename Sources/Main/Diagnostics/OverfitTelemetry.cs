// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;

namespace DevOnBike.Overfit.Diagnostics
{
    public static class OverfitTelemetry
    {
        public const string MeterName = "DevOnBike.Overfit";
        public const string Version = "1.0.0";

        /// <summary>
        /// Master switch for metrics and tracing emitted through OverfitTelemetry.
        /// Keep this cheap and explicit.
        /// </summary>
        public static bool Enabled { get; set; } = true;

        /// <summary>
        /// Enables ActivitySource spans. Metrics can remain enabled while tracing is off.
        /// </summary>
        public static bool TraceEnabled { get; set; } = false;

        public static readonly Meter Meter = new(MeterName, Version);
        public static readonly ActivitySource Tracer = new(MeterName, Version);

        // Core durations
        public static readonly Histogram<double> ModuleDurationMs = Meter.CreateHistogram<double>(
            "overfit.module.duration.ms",
            unit: "ms",
            description: "Execution time of high-level modules.");

        public static readonly Histogram<double> GraphBackwardDurationMs = Meter.CreateHistogram<double>(
            "overfit.graph.backward.duration.ms",
            unit: "ms",
            description: "Backward pass duration.");

        public static readonly Histogram<double> BackwardOpDurationMs = Meter.CreateHistogram<double>(
            "overfit.backward.op.duration.ms",
            unit: "ms",
            description: "Backward opcode execution time.");

        public static readonly Histogram<double> EpochDurationMs = Meter.CreateHistogram<double>(
            "overfit.epoch.duration.ms",
            unit: "ms",
            description: "Epoch duration.");

        public static readonly Histogram<double> BatchDurationMs = Meter.CreateHistogram<double>(
            "overfit.batch.duration.ms",
            unit: "ms",
            description: "Batch duration.");

        public static readonly Histogram<double> OptimizerStepDurationMs = Meter.CreateHistogram<double>(
            "overfit.optimizer.step.duration.ms",
            unit: "ms",
            description: "Optimizer step duration.");

        public static readonly Histogram<double> LossValue = Meter.CreateHistogram<double>(
            "overfit.loss",
            unit: "{loss}",
            description: "Observed loss values.");

        // Allocation / memory
        public static readonly Histogram<long> ModuleAllocatedBytes = Meter.CreateHistogram<long>(
            "overfit.module.alloc.bytes",
            unit: "By",
            description: "Managed bytes allocated during module execution.");

        public static readonly Histogram<long> GraphAllocatedBytes = Meter.CreateHistogram<long>(
            "overfit.graph.alloc.bytes",
            unit: "By",
            description: "Managed bytes allocated during backward.");

        public static readonly Histogram<long> BackwardOpAllocatedBytes = Meter.CreateHistogram<long>(
            "overfit.backward.op.alloc.bytes",
            unit: "By",
            description: "Managed bytes allocated by a backward opcode.");

        public static readonly Histogram<long> AllocationBytes = Meter.CreateHistogram<long>(
            "overfit.allocation.bytes",
            unit: "By",
            description: "Reported tensor/buffer allocations.");

        public static readonly UpDownCounter<long> NativeMemoryBytes = Meter.CreateUpDownCounter<long>(
            "overfit.memory.native.bytes",
            unit: "By",
            description: "Native / unmanaged memory tracked by Overfit.");

        // Counters
        public static readonly Counter<long> ModuleCount = Meter.CreateCounter<long>(
            "overfit.module.count",
            unit: "{module}",
            description: "Number of completed module executions.");

        public static readonly Counter<long> GraphCount = Meter.CreateCounter<long>(
            "overfit.graph.count",
            unit: "{graph}",
            description: "Number of completed backward passes.");

        public static readonly Counter<long> TapeOpCount = Meter.CreateCounter<long>(
            "overfit.graph.tape_ops",
            unit: "{op}",
            description: "Tape ops completed in backward passes.");

        public static readonly Counter<long> GraphRecordTotalCount = Meter.CreateCounter<long>(
            "overfit.graph.record_op.count",
            unit: "{op}",
            description: "Graph record record total count");

        public static readonly Counter<long> BackwardOpCount = Meter.CreateCounter<long>(
            "overfit.backward.op.count",
            unit: "{op}",
            description: "Number of backward opcode executions.");

        public static readonly Counter<long> Gen0Collections = Meter.CreateCounter<long>(
            "overfit.gc.gen0.count",
            unit: "{gc}",
            description: "Gen0 collections observed by Overfit.");

        public static readonly Counter<long> Gen1Collections = Meter.CreateCounter<long>(
            "overfit.gc.gen1.count",
            unit: "{gc}",
            description: "Gen1 collections observed by Overfit.");

        public static readonly Counter<long> Gen2Collections = Meter.CreateCounter<long>(
            "overfit.gc.gen2.count",
            unit: "{gc}",
            description: "Gen2 collections observed by Overfit.");


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
        public static void RecordModule(
            string module,
            string phase,
            bool trainMode,
            TimeSpan elapsed,
            long allocatedBytes)
        {
            if (!Enabled)
            {
                return;
            }

            TagList tags = default;
            tags.Add("module", module);
            tags.Add("phase", phase);
            tags.Add("train_mode", trainMode);

            ModuleDurationMs.Record(elapsed.TotalMilliseconds, tags);
            ModuleAllocatedBytes.Record(allocatedBytes, tags);
            ModuleCount.Add(1, tags);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RecordGraphBackward(
            string phase,
            bool trainMode,
            TimeSpan elapsed,
            long allocatedBytes,
            int tapeOps,
            int gen0Collections,
            int gen1Collections,
            int gen2Collections)
        {
            if (!Enabled)
            {
                return;
            }

            TagList tags = default;
            tags.Add("phase", phase);
            tags.Add("train_mode", trainMode);

            GraphBackwardDurationMs.Record(elapsed.TotalMilliseconds, tags);
            GraphAllocatedBytes.Record(allocatedBytes, tags);
            GraphCount.Add(1, tags);
            TapeOpCount.Add(tapeOps, tags);

            if (gen0Collections != 0)
            {
                Gen0Collections.Add(gen0Collections, tags);
            }

            if (gen1Collections != 0)
            {
                Gen1Collections.Add(gen1Collections, tags);
            }

            if (gen2Collections != 0)
            {
                Gen2Collections.Add(gen2Collections, tags);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RecordBackwardOp(
            string opcode,
            TimeSpan elapsed,
            long allocatedBytes)
        {
            if (!Enabled)
            {
                return;
            }

            TagList tags = default;
            tags.Add("opcode", opcode);

            BackwardOpDurationMs.Record(elapsed.TotalMilliseconds, tags);
            BackwardOpAllocatedBytes.Record(allocatedBytes, tags);
            BackwardOpCount.Add(1, tags);
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RecordAllocation(
            string owner,
            string resourceType,
            long bytes,
            bool pooled,
            bool managed)
        {
            if (!Enabled)
            {
                return;
            }

            TagList tags = default;
            tags.Add("owner", owner);
            tags.Add("resource_type", resourceType);
            tags.Add("pooled", pooled);
            tags.Add("managed", managed);

            AllocationBytes.Record(bytes, tags);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddNativeMemory(long bytesDelta)
        {
            if (!Enabled || bytesDelta == 0)
            {
                return;
            }

            NativeMemoryBytes.Add(bytesDelta);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RecordEpoch(int epoch, TimeSpan elapsed, double loss)
        {
            if (!Enabled)
            {
                return;
            }

            TagList tags = default;
            tags.Add("epoch", epoch);

            EpochDurationMs.Record(elapsed.TotalMilliseconds, tags);
            LossValue.Record(loss, tags);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RecordBatch(bool trainMode, TimeSpan elapsed)
        {
            if (!Enabled)
            {
                return;
            }

            TagList tags = default;
            tags.Add("train_mode", trainMode);

            BatchDurationMs.Record(elapsed.TotalMilliseconds, tags);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RecordOptimizerStep(bool trainMode, TimeSpan elapsed)
        {
            if (!Enabled)
            {
                return;
            }

            TagList tags = default;
            tags.Add("train_mode", trainMode);

            OptimizerStepDurationMs.Record(elapsed.TotalMilliseconds, tags);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RecordLoss(double loss, bool trainMode)
        {
            if (!Enabled)
            {
                return;
            }

            TagList tags = default;
            tags.Add("train_mode", trainMode);

            LossValue.Record(loss, tags);
        }
    }
}
