using DevOnBike.Overfit.Diagnostics.Contracts;

namespace DevOnBike.Overfit.Diagnostics
{
    public sealed class NullOverfitDiagnosticsSink : IOverfitDiagnosticsSink
    {
        public static readonly NullOverfitDiagnosticsSink Instance = new();

        private NullOverfitDiagnosticsSink()
        {
        }

        public void OnKernelCompleted(in KernelDiagnosticEvent evt)
        {
        }

        public void OnModuleCompleted(in ModuleDiagnosticEvent evt)
        {
        }

        public void OnGraphCompleted(in GraphDiagnosticEvent evt)
        {
        }

        public void OnAllocation(in AllocationDiagnosticEvent evt)
        {
        }

        public void OnCounter(string name, long value)
        {
        }
    }
}
