// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Adaptive
{
    /// <summary>
    /// Policy for <see cref="AdaptiveAnomalyMonitor"/> — when to recommend per-pod LoRA
    /// adaptation and how to train it. Motivated by the measured deployment finding: a
    /// cross-pod base scores a single pod's BENIGN regime as elevated (a false positive
    /// in the [<see cref="AlertThreshold"/>, <see cref="CriticalThreshold"/>) band), which
    /// a cheap per-pod LM-head LoRA flattens toward zero without blinding the detector
    /// (see <c>GptAnomalyLoRATargetComparisonProductionTests</c>).
    /// </summary>
    public sealed class AdaptivePolicy
    {
        /// <summary>Rolling window (snapshots) each per-pod detector sees. Must fit the model's context.</summary>
        public int ContextSnapshots { get; init; } = 8;

        /// <summary>Score at/above which a snapshot is "elevated" (anomaly flag). Default 5.</summary>
        public float AlertThreshold { get; init; } = 5.0f;

        /// <summary>
        /// Score at/above which a snapshot is a real incident (NOT false-positive pressure,
        /// and NOT buffered as benign for adaptation). Default 10.
        /// </summary>
        public float CriticalThreshold { get; init; } = 10.0f;

        /// <summary>
        /// Consecutive elevated-but-sub-critical scores that flag the pod as needing
        /// adaptation (sustained false-positive pressure = base miscalibration for this pod,
        /// not a transient spike). Default 5.
        /// </summary>
        public int AdaptAfterStreak { get; init; } = 5;

        /// <summary>Max benign (sub-critical) snapshots retained per pod for adaptation. Default 48.</summary>
        public int BenignWindow { get; init; } = 48;

        /// <summary>Minimum buffered benign snapshots required before adaptation can run. Default 24.</summary>
        public int MinBenignWindow { get; init; } = 24;

        /// <summary>LoRA rank for per-pod adaptation. Default 16.</summary>
        public int LoRARank { get; init; } = 16;

        /// <summary>LoRA fine-tune steps. Default 300.</summary>
        public int LoRASteps { get; init; } = 300;

        /// <summary>LoRA learning rate. Default 1e-2.</summary>
        public float LoRALearningRate { get; init; } = 1e-2f;

        /// <summary>Directory where per-pod adapter <c>.bin</c> files are stored.</summary>
        public required string AdapterDirectory { get; init; }
    }
}
