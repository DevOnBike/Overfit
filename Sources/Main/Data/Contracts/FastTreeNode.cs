// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    internal struct FastTreeNode
    {
        // We use a bit flag or a special index to save space
        public bool IsLeaf;

        // Data for a decision node
        public int FeatureIndex;
        public float Threshold;

        // Data for a leaf node (regression/classification result)
        public float Value;

        // Child indices in the flat tree array (instead of references)
        // This allows the entire tree to be stored in a FastBuffer&lt;FastTreeNode&gt;
        public int LeftChildIndex;
        public int RightChildIndex;
    }
}