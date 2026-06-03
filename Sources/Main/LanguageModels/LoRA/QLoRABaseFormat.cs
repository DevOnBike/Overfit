// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    /// <summary>
    /// Quantization format for a QLoRA frozen base (<c>Gpt1LoRAFineTuner(quantizeBase: true)</c>).
    /// A precision-vs-RAM knob: <see cref="Q4K"/> saves the most RAM (~7× vs F32) at slightly lower
    /// base fidelity; <see cref="Q8"/> is higher fidelity (~3.8× vs F32) and — because its blocks are
    /// 32 elements, not 256 — works for projections whose input dim is &lt; 256 (e.g. per-head
    /// attention, dHead = 64), where Q4_K cannot.
    /// </summary>
    public enum QLoRABaseFormat
    {
        /// <summary>4.5-bit K-quant (256-element super-blocks). Default — maximum RAM saving.</summary>
        Q4K,

        /// <summary>8-bit (32-element blocks). Higher fidelity; usable below input dim 256.</summary>
        Q8,
    }
}
