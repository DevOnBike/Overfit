// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly record struct GraphDiagnosticEvent(
        int TapeOpCount,
        double BackwardMs,
        long AllocatedBytes,
        int Gen0Collections,
        int Gen1Collections,
        int Gen2Collections,
        string Phase = "backward",
        bool IsTraining = true,
        int BatchSize = 0);
}
