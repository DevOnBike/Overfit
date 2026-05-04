// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.LoRA
{

    public readonly struct LoRAOptions
    {
        public LoRAOptions(
            int rank,
            float alpha,
            float dropout,
            LoRATargetModules targetModules)
        {
            Rank = rank;
            Alpha = alpha;
            Dropout = dropout;
            TargetModules = targetModules;
        }

        public int Rank { get; }

        public float Alpha { get; }

        public float Dropout { get; }

        public LoRATargetModules TargetModules { get; }

        public float Scale => Rank == 0 ? 0f : Alpha / Rank;
    }
}
