// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    internal struct FastTreeNode
    {
        public int[] FeatureIndices;
        public float[] Thresholds;
        public float[] Values;
        public int Depth;
    }
}