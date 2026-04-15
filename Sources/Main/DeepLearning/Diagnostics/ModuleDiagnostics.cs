using System.Diagnostics;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Diagnostics.Contracts;

namespace DevOnBike.Overfit.DeepLearning.Diagnostics
{
    internal static class ModuleDiagnostics
    {
        public static ModuleExecutionContext Begin(
            string moduleType,
            string phase,
            bool isTraining,
            int batchSize = 0,
            int inputRows = 0,
            int inputCols = 0,
            int outputRows = 0,
            int outputCols = 0)
        {
            if (!OverfitDiagnostics.IsEnabled())
            {
                return default;
            }

            return new ModuleExecutionContext(
            moduleType,
            phase,
            isTraining,
            batchSize,
            inputRows,
            inputCols,
            outputRows,
            outputCols,
            GC.GetTotalAllocatedBytes(false),
            Stopwatch.GetTimestamp());
        }

        public static void End(in ModuleExecutionContext ctx)
        {
            if (!ctx.IsEnabled)
            {
                return;
            }

            var endAlloc = GC.GetTotalAllocatedBytes(false);
            var endTicks = Stopwatch.GetTimestamp();

            OverfitDiagnostics.ModuleCompleted(new ModuleDiagnosticEvent(
            ModuleType: ctx.ModuleType,
            Phase: ctx.Phase,
            DurationMs: (endTicks - ctx.StartTimestamp) * 1000.0 / Stopwatch.Frequency,
            InputRows: ctx.InputRows,
            InputCols: ctx.InputCols,
            AllocatedBytes: endAlloc - ctx.StartAllocatedBytes,
            IsTraining: ctx.IsTraining,
            BatchSize: ctx.BatchSize,
            OutputRows: ctx.OutputRows,
            OutputCols: ctx.OutputCols));
        }

        internal readonly record struct ModuleExecutionContext(
            string ModuleType,
            string Phase,
            bool IsTraining,
            int BatchSize,
            int InputRows,
            int InputCols,
            int OutputRows,
            int OutputCols,
            long StartAllocatedBytes,
            long StartTimestamp)
        {
            public bool IsEnabled => !string.IsNullOrEmpty(ModuleType);
        }
    }
}
