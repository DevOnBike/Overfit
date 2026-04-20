// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Demo.Unity.Server
{
    /// <summary>
    ///     Immutable configuration for the swarm demo: environment geometry, physics,
    ///     fitness shaping, network architecture and evolutionary-strategy hyperparameters.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Grouped as a single record so the trainer, the environment and the Unity
    ///         server all operate from the same snapshot — no static constants scattered
    ///         across files, and every call site can see which parameters it reads.
    ///     </para>
    ///     <para>
    ///         <see cref="Default"/> provides the tuned settings used in the showcase:
    ///         a population of 256 genomes evaluated on 100 000 simultaneous bots, trained
    ///         with OpenAI-style Evolution Strategies for 200 generations. On a Ryzen 9
    ///         9950X3D the default run finishes in roughly 3–5 minutes.
    ///     </para>
    /// </remarks>
    public sealed record SwarmConfig(
        // ── Simulation layout ────────────────────────────────────────────
        int SwarmSize,
        int InputSize,
        int OutputSize,
        int HiddenSize,
        int Port,

        // ── Physics (must stay aligned with the Unity client) ────────────
        float DeltaTime,
        float AccelerationScale,
        float Damping,
        float ArenaSize,
        float PredatorRadius,
        float PredatorFearRange,
        float OrbitRadius,
        float OrbitTolerance,

        // ── Fitness shaping ──────────────────────────────────────────────
        int FramesPerGeneration,
        float PredatorPenalty,
        float ArenaPenalty,
        float OrbitRewardScale,
        float TangentBonusScale,

        // ── Evolution Strategies hyperparameters ─────────────────────────
        int PopulationSize,
        float Sigma,
        float LearningRate,
        int NoiseTableLength,
        int? Seed)
    {
        /// <summary>
        ///     Number of trainable parameters in the default MLP (4→16→2):
        ///     (4*16 + 16) weights and biases for the hidden layer + (16*2 + 2) for the output.
        ///     Equivalent to <see cref="SwarmBrain.CountParameters"/> but exposed here so the
        ///     trainer can size genome buffers before constructing a brain.
        /// </summary>
        public int GenomeSize => SwarmBrain.CountParameters(InputSize, HiddenSize, OutputSize);

        public static SwarmConfig Default => new(
            // 102 400 = 256 × 400 — cohort size of 400 bots per genome gives a stable fitness
            // estimator while still rounding to "about 100k bots" for the visible swarm.
            SwarmSize: 102_400,
            InputSize: 4,
            OutputSize: 2,
            HiddenSize: 16,
            Port: 5000,

            DeltaTime: 1f / 60f,
            AccelerationScale: 35f,
            Damping: 0.92f,
            ArenaSize: 30f,
            PredatorRadius: 2.5f,
            PredatorFearRange: 10f,
            OrbitRadius: 3f,
            OrbitTolerance: 1f,

            FramesPerGeneration: 600,
            PredatorPenalty: 20f,
            ArenaPenalty: 10f,
            OrbitRewardScale: 2f,
            TangentBonusScale: 0.2f,

            // With a 114-parameter brain and 256 candidates per generation, ES is the clear
            // choice (6–14× faster than GA at this genome size; see benchmark in the README).
            PopulationSize: 256,
            Sigma: 0.1f,
            LearningRate: 0.02f,
            NoiseTableLength: 1 << 20, // 1M floats = 4 MB
            Seed: 42);
    }
}