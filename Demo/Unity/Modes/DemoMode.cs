// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;

namespace DevOnBike.Overfit.Demo.Unity.Server.Modes
{
    /// <summary>
    ///     Loads a trained brain from disk, clones it across the entire swarm, and serves
    ///     bot positions to a connected Unity client each frame. Unity drives the target
    ///     and predator coordinates; the server is authoritative over bot state.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Every bot runs the same policy (the best-ever genome produced by training),
    ///         so what the user sees is a fair emergent test: under a single shared brain,
    ///         does the swarm converge to the learned orbit-and-avoid behaviour?
    ///     </para>
    ///     <para>
    ///         There is no training in this mode — fitness is not computed, there is no
    ///         population, and the ES trainer is not instantiated. The only Overfit feature
    ///         exercised here is <see cref="SwarmBrain.Infer"/> running at real-time frame
    ///         rate across <see cref="SwarmConfig.SwarmSize"/> bots.
    ///     </para>
    /// </remarks>
    public static class DemoMode
    {
        public static void Run(SwarmConfig config, string brainPath, CancellationToken cancellation)
        {
            Console.WriteLine("[MODE: DEMO] Loading evolved brain...");

            using var brain = new SwarmBrain(config.InputSize, config.HiddenSize, config.OutputSize);

            if (!brain.LoadFromFile(brainPath))
            {
                Console.WriteLine($"[WARN] No usable brain at {brainPath}. Run offline training first.");
                Console.WriteLine("       Usage: dotnet run -- offline 200");
                return;
            }

            var environment = new SwarmEnvironment(config);
            var outputs = new float[config.SwarmSize * config.OutputSize];
            var inputs = new float[config.SwarmSize * config.InputSize];
            var rng = new Random();

            environment.Respawn(Vector2.Zero, rng);

            using var server = new UnityTcpServer(config.Port, config.SwarmSize);
            server.Start();
            Console.WriteLine($"[READY] Demo server listening on port {config.Port}. Start Unity.");

            while (!cancellation.IsCancellationRequested)
            {
                try
                {
                    server.AcceptAndServe(
                        onFrame: (target, predator) =>
                        {
                            environment.BuildInputs(target, predator, inputs);
                            RunInferenceAcrossSwarm(brain, inputs, outputs, config);
                            environment.StepDemo(outputs, target, predator, rng);
                            return environment.Positions;
                        },
                        cancellation: cancellation);
                }
                catch (Exception e) when (e is not OperationCanceledException)
                {
                    // Unity disconnecting mid-session is expected — just loop back to
                    // AcceptAndServe for a reconnection. Anything else worth logging.
                    Console.WriteLine($"[DISCONNECT] {e.GetType().Name}: {e.Message}");
                    Thread.Sleep(100);
                }
            }
        }

        /// <summary>
        ///     Single-policy inference across the entire swarm. All bots share the same brain,
        ///     so we can parallelise purely across bots with no shared-state concerns — each
        ///     bot's input/output slices are disjoint.
        /// </summary>
        private static void RunInferenceAcrossSwarm(
            SwarmBrain brain, float[] inputs, float[] outputs, SwarmConfig config)
        {
            var swarmSize = config.SwarmSize;
            var inputSize = config.InputSize;
            var outputSize = config.OutputSize;

            // SwarmBrain.Infer is NOT thread-safe (it owns the underlying Sequential's
            // PooledBuffer caches), so we serialise bot-by-bot here. This is the deliberate
            // cost of using a single shared policy instead of a per-bot cohort as in
            // training — but it's also trivially cheap at inference: 100k × tiny MLP on a
            // single thread finishes within a 60-FPS frame budget on modern CPUs.
            //
            // If frame rate becomes a concern, the fix is to hold a small pool of brain
            // clones and shard bots across them — but that complicates the demo and we
            // want to measure the single-thread story first.
            for (var i = 0; i < swarmSize; i++)
            {
                var input = new ReadOnlySpan<float>(inputs, i * inputSize, inputSize);
                var output = new Span<float>(outputs, i * outputSize, outputSize);
                brain.Infer(input, output);
            }
        }
    }
}
