// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

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
