// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics.Contracts;

namespace DevOnBike.Overfit.Diagnostics
{
    public interface IOverfitDiagnosticsSink
    {
        void OnKernelCompleted(in KernelDiagnosticEvent evt);
        void OnModuleCompleted(in ModuleDiagnosticEvent evt);
        void OnGraphCompleted(in GraphDiagnosticEvent evt);
        void OnAllocation(in AllocationDiagnosticEvent evt);
        void OnCounter(string name, long value);
    }
}
