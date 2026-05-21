// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    /// <summary>
    /// One LoRA adapter entry: which weight matrix it adapts (module + layer +
    /// head) and the trained low-rank factors. <see cref="Layer"/> is -1 for the
    /// LM head. <see cref="HeadIndex"/> is 0 for whole-matrix targets (LM head,
    /// FFN) and the per-head index for the per-head attention Q/K/V/O weights
    /// (Stage 3).
    /// </summary>
    internal readonly record struct Gpt1LoRAEntry(int Layer, LoRATargetModules Module, int HeadIndex, LoRAWeight Weight);
}
