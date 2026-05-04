// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    public interface ILoRAInjectable
    {
        bool SupportsLoRA { get; }

        ILoRAAdapter InjectLoRA(string name, in LoRAOptions options);

        bool TryGetLoRA(string name, out ILoRAAdapter? adapter);

        bool RemoveLoRA(string name);

        void FreezeBaseWeights();

        void UnfreezeBaseWeights();
    }
}
