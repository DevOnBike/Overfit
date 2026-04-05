// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    public class LayerDiagnostic
    {
        public string LayerName { get; init; }
        public int RowsBefore { get; init; }
        public int ColsBefore { get; init; }
        public int RowsAfter { get; init; }
        public int ColsAfter { get; init; }
        public long ElapsedMs { get; init; }
    }
}