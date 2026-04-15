using System.Collections.Concurrent;
using System.Diagnostics;
using DevOnBike.Overfit.Diagnostics.Contracts;

namespace DevOnBike.Overfit.Diagnostics
{
    public sealed class OpenTelemetrySink : IOverfitDiagnosticsSink
    {
        public static readonly OpenTelemetrySink Instance = new();

        private readonly ConcurrentDictionary<string, CounterState> _counters = new(StringComparer.Ordinal);

        private OpenTelemetrySink()
        {
        }

        public void OnKernelCompleted(in KernelDiagnosticEvent evt)
        {
            var tags = new TagList
            {
                { "category", evt.Category },
                { "kernel", evt.Name },
                { "phase", evt.Phase },
                { "train_mode", evt.IsTraining },
                { "batch", evt.BatchSize },
                { "features", evt.FeatureCount },
                { "input_elements", evt.InputElements },
                { "output_elements", evt.OutputElements }
            };

            OverfitTelemetry.KernelDurationMs.Record(evt.DurationMs, tags);
            OverfitTelemetry.KernelCount.Add(1, tags);
        }

        public void OnModuleCompleted(in ModuleDiagnosticEvent evt)
        {
            var tags = new TagList
            {
                { "module", evt.ModuleType },
                { "phase", evt.Phase },
                { "train_mode", evt.IsTraining },
                { "batch", evt.BatchSize },
                { "input_rows", evt.InputRows },
                { "input_cols", evt.InputCols },
                { "output_rows", evt.OutputRows },
                { "output_cols", evt.OutputCols }
            };

            OverfitTelemetry.ModuleDurationMs.Record(evt.DurationMs, tags);
            OverfitTelemetry.ModuleAllocatedBytes.Record(evt.AllocatedBytes, tags);
            OverfitTelemetry.ModuleCount.Add(1, tags);
        }

        public void OnGraphCompleted(in GraphDiagnosticEvent evt)
        {
            var tags = new TagList
            {
                { "phase", evt.Phase },
                { "train_mode", evt.IsTraining },
                { "tape_ops", evt.TapeOpCount },
                { "batch", evt.BatchSize }
            };

            OverfitTelemetry.GraphBackwardDurationMs.Record(evt.BackwardMs, tags);
            OverfitTelemetry.GraphAllocatedBytes.Record(evt.AllocatedBytes, tags);
            OverfitTelemetry.TapeOpCount.Add(evt.TapeOpCount, tags);
            OverfitTelemetry.GraphCount.Add(1, tags);
        }

        public void OnAllocation(in AllocationDiagnosticEvent evt)
        {
            var tags = new TagList
            {
                { "owner", evt.Owner },
                { "resource_type", evt.ResourceType },
                { "pooled", evt.IsPooled },
                { "managed", evt.IsManaged },
                { "elements", evt.Elements }
            };

            OverfitTelemetry.AllocationBytes.Record(evt.Bytes, tags);

            if (!evt.IsManaged)
            {
                OverfitTelemetry.NativeMemoryBytes.Add(evt.Bytes, tags);
            }
        }

        public void OnCounter(string name, long value)
        {
            _counters.AddOrUpdate(
            name,
            static (_, v) => new CounterState(v),
            static (_, state, v) =>
            {
                state.Value += v;
                return state;
            },
            value);
        }

        private sealed class CounterState
        {
            public CounterState(long value)
            {
                Value = value;
            }

            public long Value { get; set; }
        }
    }
}
