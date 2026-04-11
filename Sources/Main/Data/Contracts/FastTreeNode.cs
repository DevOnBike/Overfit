// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    internal struct FastTreeNode
    {
        // Używamy flagi bitowej lub specjalnego indeksu, aby zaoszczędzić miejsce
        public bool IsLeaf;

        // Dane dla węzła decyzyjnego
        public int FeatureIndex;
        public float Threshold;

        // Dane dla liścia (wynik regresji/klasyfikacji)
        public float Value;

        // Indeksy dzieci w płaskiej tablicy drzewa (zamiast referencji)
        // Pozwala to na trzymanie całego drzewa w FastBuffer<FastTreeNode>
        public int LeftChildIndex;
        public int RightChildIndex;
    }
}