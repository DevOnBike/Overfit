// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly record struct AllocationDiagnosticEvent(
        string Owner,
        string ResourceType,
        int Elements,
        long Bytes,
        bool IsPooled = false,
        bool IsManaged = true);
}