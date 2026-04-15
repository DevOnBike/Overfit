using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace DevOnBike.Overfit.Diagnostics
{
    public static class OverfitTelemetry
    {
        public const string MeterName = "DevOnBike.Overfit";
        public const string Version = "1.0.0";

        public static readonly Meter Meter = new(MeterName, Version);
        public static readonly ActivitySource Tracer = new(MeterName, Version);

        public static readonly Histogram<double> KernelDurationMs =
            Meter.CreateHistogram<double>(
            "overfit.kernel.duration.ms",
            unit: "ms",
            description: "Execution time of low-level kernels.");

        public static readonly Histogram<double> ModuleDurationMs =
            Meter.CreateHistogram<double>(
            "overfit.module.duration.ms",
            unit: "ms",
            description: "Execution time of high-level DL modules.");

        public static readonly Histogram<double> GraphBackwardDurationMs =
            Meter.CreateHistogram<double>(
            "overfit.graph.backward.duration.ms",
            unit: "ms",
            description: "Backward pass duration.");

        public static readonly Histogram<long> ModuleAllocatedBytes =
            Meter.CreateHistogram<long>(
            "overfit.module.alloc.bytes",
            unit: "By",
            description: "Allocated managed bytes captured for module execution.");

        public static readonly Histogram<long> GraphAllocatedBytes =
            Meter.CreateHistogram<long>(
            "overfit.graph.alloc.bytes",
            unit: "By",
            description: "Allocated managed bytes captured during backward.");

        public static readonly Histogram<long> AllocationBytes =
            Meter.CreateHistogram<long>(
            "overfit.allocation.bytes",
            unit: "By",
            description: "Reported tensor/buffer allocations.");

        public static readonly Counter<long> KernelCount =
            Meter.CreateCounter<long>(
            "overfit.kernel.count",
            unit: "{kernel}",
            description: "Number of kernel executions.");

        public static readonly Counter<long> ModuleCount =
            Meter.CreateCounter<long>(
            "overfit.module.count",
            unit: "{module}",
            description: "Number of module executions.");

        public static readonly Counter<long> GraphCount =
            Meter.CreateCounter<long>(
            "overfit.graph.count",
            unit: "{graph}",
            description: "Number of graph completion events.");

        public static readonly Counter<long> TapeOpCount =
            Meter.CreateCounter<long>(
            "overfit.graph.tape_ops",
            unit: "{op}",
            description: "Total number of tape ops recorded in completed graphs.");

        public static readonly UpDownCounter<long> NativeMemoryBytes =
            Meter.CreateUpDownCounter<long>(
            "overfit.memory.native.bytes",
            unit: "By",
            description: "Native / unmanaged memory tracked by Overfit diagnostics.");
    }
}
