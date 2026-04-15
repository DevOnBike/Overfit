// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly record struct ModuleDiagnosticEvent(
        string ModuleType,
        string Phase,
        double DurationMs,
        int InputRows,
        int InputCols,
        long AllocatedBytes,
        bool IsTraining = false,
        int BatchSize = 0,
        int OutputRows = 0,
        int OutputCols = 0);
}
