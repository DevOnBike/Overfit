// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    public interface ILoRAAdapter : IDisposable
    {
        string Name { get; }

        LoRAOptions Options { get; }

        long TrainableParameterCount { get; }

        bool IsEnabled { get; }

        void Enable();

        void Disable();

        void Save(string path);

        void Load(string path);

        void ZeroGrad();
    }
}
