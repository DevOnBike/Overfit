// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Demo.Unity.Server.Modes;

namespace DevOnBike.Overfit.Demo.Unity.Server
{
    /// <summary>
    ///     Entry point for the Unity swarm showcase. Two modes:
    ///     <list type="bullet">
    ///         <item><c>offline [generations]</c> — train from scratch (or resume from a
    ///               checkpoint) with no Unity attached. Fastest way to get a brain.</item>
    ///         <item><c>demo</c> — load the trained brain and serve bot positions to a
    ///               connected Unity client. Target and predator are Unity-driven.</item>
    ///     </list>
    /// </summary>
    /// <remarks>
    ///     A typical workflow: <c>dotnet run -- offline 200</c> to train, then
    ///     <c>dotnet run -- demo</c> to watch the policy in Unity.
    /// </remarks>
    internal static class Program
    {
        private static readonly string BrainPath = Path.Combine(AppContext.BaseDirectory, "swarm_brain.bin");
        private static readonly string CheckpointPath = Path.Combine(AppContext.BaseDirectory, "swarm_checkpoint.bin");

        private static void Main(string[] args)
        {
            Console.WriteLine("=== Overfit Swarm Engine ===");

            var mode = args.Length > 0 ? args[0].ToLowerInvariant() : "demo";
            var config = SwarmConfig.Default;

            using var cts = new CancellationTokenSource();
            SetupShutdown(cts);

            switch (mode)
            {
                case "offline":
                {
                    var generations = args.Length > 1 && int.TryParse(args[1], out var g) ? g : 200;
                    OfflineTrainingMode.Run(config, BrainPath, CheckpointPath, generations, cts.Token);
                    break;
                }

                case "demo":
                {
                    DemoMode.Run(config, BrainPath, cts.Token);
                    break;
                }

                default:
                {
                    Console.Error.WriteLine($"Unknown mode: {mode}.");
                    Console.Error.WriteLine("Usage:");
                    Console.Error.WriteLine("  offline [generations]   Train without Unity (default: 200)");
                    Console.Error.WriteLine("  demo                    Serve a trained brain to Unity");
                    Environment.ExitCode = 1;
                    break;
                }
            }
        }

        private static void SetupShutdown(CancellationTokenSource cts)
        {
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
                Console.WriteLine("\n[SHUTDOWN] Stopping (saving checkpoint)...");
            };
        }
    }
}
