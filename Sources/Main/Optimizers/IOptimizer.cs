// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Optimizers
{
    public interface IOptimizer
    {
        float LearningRate { get; set; }

        // Krok optymalizacji (aktualizacja wag)
        void Step();

        // Czyszczenie gradientów przed nowym batchem
        void ZeroGrad();
    }
}