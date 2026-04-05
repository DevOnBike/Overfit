// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    public class TableSchema
    {
        public List<ColumnDefinition> Features { get; set; } = [];
        public ColumnDefinition Target { get; set; }
    }
}