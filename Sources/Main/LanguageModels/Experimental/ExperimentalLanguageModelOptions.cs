// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Experimental
{
    /// <summary>
    /// Experimental language-model switches.
    ///
    /// Defaults must keep the stable/reference path unchanged.
    /// These options are intended for manual experiments, demos, and
    /// benchmarks while GPT training/runtime APIs are being stabilized.
    /// </summary>
    public static class ExperimentalLanguageModelOptions
    {
        // Default ON: the parallel SDPA backward is bit-identical to sequential
        // (each batch index runs the same kernel; Parallel.For only splits the
        // batch loop — no cross-batch reduction). Author observed ~27% backward
        // speedup on data-parallel TinyShakespeare. Flipped from experimental
        // to default ON during the CPU-saturation pass.
        private volatile static bool _enableParallelAttentionBackward = true;

        /// <summary>
        /// Enables the parallel implementation of
        /// <c>ScaledDotProductAttentionBackward</c>.
        ///
        /// Default: <c>true</c>. Parallel implementation is bit-identical to
        /// sequential — each batch goes through the same per-batch kernel,
        /// only the outer batch loop is parallelized. No determinism risk from
        /// reduction order.
        ///
        /// Set to <c>false</c> only when debugging suspected parallelism bugs
        /// or when running with batch size 1 (where parallel-over-batch is a
        /// no-op and the per-call Parallel.For overhead is pure waste).
        /// </summary>
        public static bool EnableParallelAttentionBackward
        {
            get => _enableParallelAttentionBackward;
            set => _enableParallelAttentionBackward = value;
        }
    }
}
