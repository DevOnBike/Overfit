// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Demo.Unity.Server.Modes
{
    /// <summary>
    ///     Pure-CPU training without a Unity client attached. Trains the swarm against a
    ///     scripted environment (target teleports every 150 frames, predator orbits origin
    ///     slowly) for a configurable number of generations and writes the best brain to
    ///     disk for later use in <see cref="DemoMode"/>.
    /// </summary>
    /// <remarks>
    ///     This is the fastest way to get from zero to a trained brain — with no rendering
    ///     bottleneck the trainer runs at full CPU. Typical numbers on a Ryzen 9 9950X3D:
    ///     200 generations of the default config finish in 3–5 minutes, producing a policy
    ///     that orbits the target while avoiding the predator.
    /// </remarks>
    public static class OfflineTrainingMode
    {
        public static void Run(SwarmConfig config, string brainPath, string checkpointPath, int generations, CancellationToken cancellation)
        {
            Console.WriteLine($"[MODE: OFFLINE] Training for {generations} generations.");
            Console.WriteLine(
                $"    population={config.PopulationSize}, genome={config.GenomeSize}, " +
                $"swarm={config.SwarmSize}, bots/genome={config.SwarmSize / config.PopulationSize}");

            using var trainer = new SwarmTrainer(config, brainPath, checkpointPath);
            trainer.Train(generations, cancellation);
        }
    }
}
