// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    public class ColumnDefinition
    {
        public string Name { get; set; }
        public ColumnType Type { get; set; }
    }
    
    internal struct FastTree
    {
        public int[] FeatureIndices;
        public float[] Thresholds;
        public float[] Values; // Średnia wartość w liściu (dla regresji ceny)
        public int Depth;
    }

}
