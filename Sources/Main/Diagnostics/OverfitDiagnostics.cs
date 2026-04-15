using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Diagnostics
{
    public static class OverfitDiagnostics
    {
        public static bool Enabled { get; set; }

        public static IOverfitDiagnosticsSink Sink { get; set; } = NullOverfitDiagnosticsSink.Instance;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsEnabled() => Enabled && Sink is not NullOverfitDiagnosticsSink;
    }

}