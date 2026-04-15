using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Diagnostics.Contracts;

namespace DevOnBike.Overfit.Diagnostics
{
    public static class OverfitDiagnostics
    {
        private static IOverfitDiagnosticsSink _sink = NullOverfitDiagnosticsSink.Instance;

        public static bool Enabled { get; set; }

        public static IOverfitDiagnosticsSink Sink
        {
            get => _sink;
            set => _sink = value ?? NullOverfitDiagnosticsSink.Instance;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsEnabled()
        {
            return Enabled && _sink is not NullOverfitDiagnosticsSink;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void KernelCompleted(in KernelDiagnosticEvent evt)
        {
            if (IsEnabled())
            {
                _sink.OnKernelCompleted(evt);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ModuleCompleted(in ModuleDiagnosticEvent evt)
        {
            if (IsEnabled())
            {
                _sink.OnModuleCompleted(evt);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void GraphCompleted(in GraphDiagnosticEvent evt)
        {
            if (IsEnabled())
            {
                _sink.OnGraphCompleted(evt);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Allocation(in AllocationDiagnosticEvent evt)
        {
            if (IsEnabled())
            {
                _sink.OnAllocation(evt);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Counter(string name, long value)
        {
            if (IsEnabled())
            {
                _sink.OnCounter(name, value);
            }
        }
    }
}
