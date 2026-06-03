// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    /// <summary>
    /// Configuration for <see cref="QLoRAFineTuner"/>. Defaults are the empirically validated
    /// known-good settings for fine-tuning a quantized Qwen/Llama GGUF on CPU (see ROADMAP / the
    /// knowledge-injection demo). Most callers only tweak <see cref="Epochs"/> and <see cref="Rank"/>.
    /// </summary>
    public sealed class QLoRAOptions
    {
        /// <summary>LoRA rank (adapter capacity). 8 is a good default; higher = more capacity + larger adapter.</summary>
        public int Rank { get; init; } = 8;

        /// <summary>Number of passes over the text. 1 = light adaptation; 3–5 = solid learning; more = memorization.</summary>
        public int Epochs { get; init; } = 3;

        /// <summary>Tokens per training sequence — the text is chunked into windows of this length.
        /// Larger = more context per step but more RAM/time (training cost is ~linear in this).</summary>
        public int ChunkLength { get; init; } = 256;

        /// <summary>Adam learning rate. 0.002 is the validated value (the knowledge-injection demo);
        /// 0.002–0.005 is the stable range.</summary>
        public float LearningRate { get; init; } = 0.002f;

        /// <summary>Global gradient-norm clip — keeps the high-variance LM-head LoRA stable.</summary>
        public float GradientClipNorm { get; init; } = 0.5f;

        /// <summary>Add a LoRA adapter on the LM head too (direct output capacity — needed to recite
        /// specific new tokens; high-variance, which is why the clip + epsilon below matter).</summary>
        public bool LoRAOnLmHead { get; init; } = true;

        /// <summary>RNG seed for LoRA initialization (reproducibility).</summary>
        public int Seed { get; init; } = 1;

        /// <summary>
        /// Adam epsilon. KEPT AT 1e-4 (not the 1e-8 default) on purpose: on a small overfit set the loss
        /// gets very low and the 1e-8 default makes Adam's 1/sqrt(v) term blow up catastrophically. This
        /// is the single most important non-obvious setting — change it only if you know why.
        /// </summary>
        public float AdamEpsilon { get; init; } = 1e-4f;
    }
}
