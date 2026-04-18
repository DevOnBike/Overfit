// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Numerics;
using System.Runtime.InteropServices;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Demo.Unity.Server
{
    /// <summary>
    /// Overfit Swarm Engine — unified trainer + demo server for Unity.
    ///
    /// Architecture: server is AUTHORITY on bot positions.
    /// Unity sends only target/predator coords, server runs physics and returns positions.
    /// Benefits:
    ///   - Server can detect predator collisions (survival-based fitness)
    ///   - Proper orbit-around-target evaluation
    ///   - 50% reduction in network transfer (no per-bot inputs to send)
    ///
    /// Protocol (all floats little-endian):
    ///   Unity → Server: [targetX, targetZ, predatorX, predatorZ] = 16 bytes
    ///   Server → Unity: [posX, posZ] * SwarmSize = 800 KB at 100k
    ///
    /// Brain model (10 floats):
    ///   W: 2×4 matrix + b: 2
    ///   Forward: out = tanh(W @ normalize_inputs(bot_state) + b)
    ///   Inputs: [target_dx_n, target_dz_n, fear_x, fear_z]  (normalized)
    ///   Outputs: [accel_x, accel_z] ∈ [-1, 1] after tanh
    /// </summary>
    internal static class Program
    {
        // ================================================================
        // CONFIG
        // ================================================================

        private const int SwarmSize = 100_000;
        private const int InputSize = 4;
        private const int OutputSize = 2;
        private const int GenomeSize = InputSize * OutputSize + OutputSize; // 8 + 2 = 10
        private const int Port = 5000;
        private const int CommandSize = 4 * sizeof(float);

        // Physics constants (must match Unity expectations)
        private const float DeltaTime = 1f / 60f;
        private const float AccelerationScale = 35f;
        private const float Damping = 0.92f;
        private const float ArenaSize = 30f;
        private const float PredatorRadius = 2.5f;
        private const float PredatorFearRange = 10f;
        private const float OrbitRadius = 3f;
        private const float OrbitTolerance = 1f;

        // GA config
        private const int FramesPerGeneration = 600; // 10s at 60fps - more time to evaluate
        private const float ElitePercent = 0.15f;     // 15% elite - more stable selection
        private const float MutationRate = 0.08f;     // 8% large mutations
        private const float MutationSmall = 0.01f;    // smaller small mutations - exploit when close
        private const float MutationLarge = 0.1f;
        private const float WeightClamp = 2.5f;

        private static readonly string BrainPath = Path.Combine(AppContext.BaseDirectory, "swarm_brain.bin");
        private static readonly string PopulationPath = Path.Combine(AppContext.BaseDirectory, "swarm_population.bin");

        // ================================================================
        // ENTRY
        // ================================================================

        private static void Main(string[] args)
        {
            var mode = args.Length > 0 ? args[0].ToLowerInvariant() : "demo";
            Console.WriteLine("=== Overfit Swarm Engine ===");

            mode = "demo"; // TEMP OVERRIDE - comment out to enable command-line mode selection

            switch (mode)
            {
                case "training" or "train":
                    RunTraining();
                    break;
                case "offline":
                    RunOfflineTraining(args);
                    break;
                case "demo":
                    RunDemo();
                    break;
                default:
                    Console.Error.WriteLine($"Unknown mode: {mode}.");
                    Console.Error.WriteLine("Usage:");
                    Console.Error.WriteLine("  training  - GA training with Unity client (60 fps cap)");
                    Console.Error.WriteLine("  offline   - GA training without Unity (MUCH faster)");
                    Console.Error.WriteLine("              optional: offline <generations> (default 200)");
                    Console.Error.WriteLine("  demo      - Load brain and serve to Unity");
                    break;
            }
        }

        // ================================================================
        // OFFLINE TRAINING — no Unity, max speed
        // Runs simulated environment server-side with random target teleports.
        // Typically 50-100x faster than online training with Unity.
        // ================================================================

        private static void RunOfflineTraining(string[] args)
        {
            var generations = args.Length > 1 && int.TryParse(args[1], out var g) ? g : 200;
            Console.WriteLine($"[MODE: OFFLINE] Training for {generations} generations (no Unity).");

            using var populationTensor = new FastTensor<float>(SwarmSize, GenomeSize, clearMemory: false);
            var population = populationTensor.GetView().AsSpan();

            var fitness = new float[SwarmSize];
            var indices = new int[SwarmSize];
            for (var i = 0; i < SwarmSize; i++) { indices[i] = i; }

            var rng = new Random(42);
            InitPopulation(population, rng);
            TryLoadPopulation(populationTensor);

            var positions = new Vector2[SwarmSize];
            var velocities = new Vector2[SwarmSize];

            // Simulated environment - target teleports randomly, predator orbits slowly
            var target = new Vector2(0, 0);
            var predator = new Vector2(3, 3);
            var predatorAngle = 0f;

            SpawnAll(positions, velocities, target, rng);

            using var inputsTensor = new FastTensor<float>(SwarmSize, InputSize, clearMemory: false);
            using var outputsTensor = new FastTensor<float>(SwarmSize, OutputSize, clearMemory: false);

            var startGen = LoadGeneration();
            var swTotal = Stopwatch.StartNew();

            using var cts = new CancellationTokenSource();
            SetupShutdown(cts);

            try
            {
                for (var gen = startGen; gen < startGen + generations && !cts.IsCancellationRequested; gen++)
                {
                    var swGen = Stopwatch.StartNew();

                    for (var frame = 0; frame < FramesPerGeneration && !cts.IsCancellationRequested; frame++)
                    {
                        // Teleport target every 150 frames (2.5s equivalent) - give bots time to reach
                        if (frame % 150 == 0)
                        {
                            target = new Vector2(
                                (float)(rng.NextDouble() * 24 - 12),  // smaller range -12..+12
                                (float)(rng.NextDouble() * 24 - 12));
                        }

                        // Predator slowly orbits origin (very slow - easier to avoid)
                        predatorAngle += 0.005f;  // was 0.02 - too fast
                        predator = new Vector2(
                            MathF.Cos(predatorAngle) * 10f,
                            MathF.Sin(predatorAngle) * 10f);

                        BuildInputs(positions, target, predator, inputsTensor.GetView().AsSpan());
                        InferAllBots(population, inputsTensor.GetView().AsReadOnlySpan(), outputsTensor.GetView().AsSpan());
                        StepPhysicsAndFitness(positions, velocities,
                            outputsTensor.GetView().AsReadOnlySpan(), target, predator, fitness, rng);
                    }

                    var (bestFitness, avgFitness) = DoGenerationStep(population, fitness, indices, rng);

                    SaveBrain(population, indices[0]);

                    if ((gen + 1) % 10 == 0)
                    {
                        SavePopulationFromSpan(population);
                        SaveGeneration(gen + 1);
                    }

                    var eliteBase = indices[0] * GenomeSize;
                    var genTime = swGen.Elapsed.TotalSeconds;
                    var totalTime = swTotal.Elapsed.TotalSeconds;
                    var eta = (generations - (gen - startGen + 1)) * genTime;

                    Console.WriteLine(
                        $"[GEN {gen + 1,4}/{startGen + generations}] " +
                        $"best={bestFitness,7:F1}  avg={avgFitness,7:F1}  " +
                        $"W=[{population[eliteBase + 0]:F2},{population[eliteBase + 1]:F2}; " +
                        $"{population[eliteBase + 4]:F2},{population[eliteBase + 5]:F2}]  " +
                        $"fear=[{population[eliteBase + 2]:F2},{population[eliteBase + 7]:F2}]  " +
                        $"gen={genTime:F1}s  ETA={eta / 60:F1}min");

                    Array.Clear(fitness);
                    SpawnAll(positions, velocities, target, rng);
                }
            }
            finally
            {
                SavePopulationFromSpan(population);
                SaveGeneration(startGen + generations);
                Console.WriteLine($"[DONE] Total time: {swTotal.Elapsed.TotalMinutes:F1} min");
                Console.WriteLine($"[DONE] Best brain saved to: {BrainPath}");
                Console.WriteLine("[DONE] Run with 'demo' mode to see results in Unity.");
            }
        }

        // ================================================================
        // TRAINING MODE — Genetic Algorithm (per-bot brain)
        // ================================================================

        private static void RunTraining()
        {
            Console.WriteLine("[MODE: TRAINING] Initializing GA...");

            using var populationTensor = new FastTensor<float>(SwarmSize, GenomeSize, clearMemory: false);
            var population = populationTensor.GetView().AsSpan();

            var fitness = new float[SwarmSize];
            var indices = new int[SwarmSize];
            for (var i = 0; i < SwarmSize; i++)
            {
                indices[i] = i;
            }

            var rng = new Random(42);
            InitPopulation(population, rng);
            TryLoadPopulation(populationTensor);

            var positions = new Vector2[SwarmSize];
            var velocities = new Vector2[SwarmSize];
            SpawnAll(positions, velocities, Vector2.Zero, rng);

            using var inputsTensor = new FastTensor<float>(SwarmSize, InputSize, clearMemory: false);
            using var outputsTensor = new FastTensor<float>(SwarmSize, OutputSize, clearMemory: false);

            using var cts = new CancellationTokenSource();
            SetupShutdown(cts);

            var listener = new TcpListener(IPAddress.Loopback, Port);
            listener.Start();
            Console.WriteLine($"[READY] Training server on port {Port}. Start Unity.");

            try
            {
                while (!cts.IsCancellationRequested)
                {
                    try
                    {
                        using var client = listener.AcceptTcpClient();
                        client.NoDelay = true;
                        using var stream = client.GetStream();
                        Console.WriteLine("[CONNECT] Unity attached. Training begins.");

                        TrainingLoop(stream, population, fitness, indices,
                            positions, velocities, inputsTensor, outputsTensor, rng, cts.Token);
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        Console.WriteLine($"[DISCONNECT] {e.GetType().Name}: {e.Message}");
                        Thread.Sleep(100);
                    }
                }
            }
            finally
            {
                listener.Stop();
                SavePopulation(populationTensor);
                Console.WriteLine("[SAVED] Final population checkpoint written.");
            }
        }

        private static void TrainingLoop(
            NetworkStream stream, Span<float> population, float[] fitness, int[] indices,
            Vector2[] positions, Vector2[] velocities,
            FastTensor<float> inputsTensor, FastTensor<float> outputsTensor,
            Random rng, CancellationToken ct)
        {
            var commandBuf = new byte[CommandSize];
            var posBuf = new byte[SwarmSize * 2 * sizeof(float)];
            var generation = LoadGeneration();
            var framesInGen = 0;
            var sw = Stopwatch.StartNew();

            while (!ct.IsCancellationRequested)
            {
                ReadExactly(stream, commandBuf);

                var target = new Vector2(
                    BinaryPrimitives.ReadSingleLittleEndian(commandBuf.AsSpan(0)),
                    BinaryPrimitives.ReadSingleLittleEndian(commandBuf.AsSpan(4)));
                var predator = new Vector2(
                    BinaryPrimitives.ReadSingleLittleEndian(commandBuf.AsSpan(8)),
                    BinaryPrimitives.ReadSingleLittleEndian(commandBuf.AsSpan(12)));

                BuildInputs(positions, target, predator, inputsTensor.GetView().AsSpan());
                InferAllBots(population, inputsTensor.GetView().AsReadOnlySpan(), outputsTensor.GetView().AsSpan());

                var outputs = outputsTensor.GetView().AsReadOnlySpan();
                StepPhysicsAndFitness(positions, velocities, outputs, target, predator, fitness, rng);

                WritePositions(positions, posBuf);
                stream.Write(posBuf);

                framesInGen++;

                if (framesInGen >= FramesPerGeneration)
                {
                    var (bestFitness, avgFitness) = DoGenerationStep(population, fitness, indices, rng);
                    generation++;
                    framesInGen = 0;

                    SaveBrain(population, indices[0]);

                    if (generation % 10 == 0)
                    {
                        SavePopulationFromSpan(population);
                        SaveGeneration(generation);
                    }

                    var eliteBase = indices[0] * GenomeSize;
                    var elapsed = sw.Elapsed.TotalSeconds;
                    Console.WriteLine(
                        $"[GEN {generation,4}] best={bestFitness,7:F1}  avg={avgFitness,7:F1}  " +
                        $"W=[{population[eliteBase + 0]:F2},{population[eliteBase + 1]:F2}; {population[eliteBase + 4]:F2},{population[eliteBase + 5]:F2}]  " +
                        $"fear=[{population[eliteBase + 2]:F2},{population[eliteBase + 7]:F2}]  t={elapsed:F1}s");

                    Array.Clear(fitness);
                    SpawnAll(positions, velocities, target, rng);
                    sw.Restart();
                }
            }
        }

        // ================================================================
        // DEMO MODE — load best brain, apply to all bots
        // ================================================================

        private static void RunDemo()
        {
            Console.WriteLine("[MODE: DEMO] Loading evolved brain...");

            var brain = LoadBrain();
            if (brain == null)
            {
                Console.WriteLine("[WARN] No brain file found. Using fallback (target attraction).");
                brain = new float[GenomeSize];
                brain[0] = 0.8f;
                brain[5] = 0.8f;
            }

            Console.WriteLine(
                $"     W=[{brain[0]:F2},{brain[1]:F2},{brain[2]:F2},{brain[3]:F2}; " +
                $"{brain[4]:F2},{brain[5]:F2},{brain[6]:F2},{brain[7]:F2}]  " +
                $"b=[{brain[8]:F2},{brain[9]:F2}]");

            using var populationTensor = new FastTensor<float>(SwarmSize, GenomeSize, clearMemory: false);
            var population = populationTensor.GetView().AsSpan();
            for (var i = 0; i < SwarmSize; i++)
            {
                brain.AsSpan().CopyTo(population.Slice(i * GenomeSize, GenomeSize));
            }

            var positions = new Vector2[SwarmSize];
            var velocities = new Vector2[SwarmSize];
            var rng = new Random();
            SpawnAll(positions, velocities, Vector2.Zero, rng);

            using var inputsTensor = new FastTensor<float>(SwarmSize, InputSize, clearMemory: false);
            using var outputsTensor = new FastTensor<float>(SwarmSize, OutputSize, clearMemory: false);

            using var cts = new CancellationTokenSource();
            SetupShutdown(cts);

            var listener = new TcpListener(IPAddress.Loopback, Port);
            listener.Start();
            Console.WriteLine($"[READY] Demo server on port {Port}. Start Unity.");

            try
            {
                while (!cts.IsCancellationRequested)
                {
                    try
                    {
                        using var client = listener.AcceptTcpClient();
                        client.NoDelay = true;
                        using var stream = client.GetStream();
                        Console.WriteLine("[CONNECT] Unity attached. Demoing evolved swarm...");

                        DemoLoop(stream, population, positions, velocities, inputsTensor, outputsTensor, rng, cts.Token);
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        Console.WriteLine($"[DISCONNECT] {e.GetType().Name}: {e.Message}");
                        Thread.Sleep(100);
                    }
                }
            }
            finally
            {
                listener.Stop();
            }
        }

        private static void DemoLoop(
            NetworkStream stream, Span<float> population,
            Vector2[] positions, Vector2[] velocities,
            FastTensor<float> inputsTensor, FastTensor<float> outputsTensor,
            Random rng, CancellationToken ct)
        {
            var commandBuf = new byte[CommandSize];
            var posBuf = new byte[SwarmSize * 2 * sizeof(float)];
            var frame = 0;
            var sw = Stopwatch.StartNew();

            while (!ct.IsCancellationRequested)
            {
                ReadExactly(stream, commandBuf);

                var target = new Vector2(
                    BinaryPrimitives.ReadSingleLittleEndian(commandBuf.AsSpan(0)),
                    BinaryPrimitives.ReadSingleLittleEndian(commandBuf.AsSpan(4)));
                var predator = new Vector2(
                    BinaryPrimitives.ReadSingleLittleEndian(commandBuf.AsSpan(8)),
                    BinaryPrimitives.ReadSingleLittleEndian(commandBuf.AsSpan(12)));

                BuildInputs(positions, target, predator, inputsTensor.GetView().AsSpan());
                InferAllBots(population, inputsTensor.GetView().AsReadOnlySpan(), outputsTensor.GetView().AsSpan());

                var outputs = outputsTensor.GetView().AsReadOnlySpan();
                StepPhysicsDemo(positions, velocities, outputs, target, predator, rng);

                WritePositions(positions, posBuf);
                stream.Write(posBuf);

                frame++;
                if (frame % 300 == 0)
                {
                    var fps = frame / sw.Elapsed.TotalSeconds;
                    Console.WriteLine($"[DEMO] frame {frame}, server {fps:F1} fps");
                }
            }
        }

        // ================================================================
        // CORE LOGIC
        // ================================================================

        private static void BuildInputs(Vector2[] positions, Vector2 target, Vector2 predator, Span<float> inputs)
        {
            // Parallel across all 100k bots - embarassingly parallel, no shared state
            // Cannot capture Span<float> in lambda, so wrap via unsafe pointer
            unsafe
            {
                fixed (float* inputsPtr = inputs)
                {
                    var ptr = (IntPtr)inputsPtr;
                    Parallel.For(0, SwarmSize, i =>
                    {
                        var inputsSpan = new Span<float>((float*)ptr, SwarmSize * InputSize);
                        var pos = positions[i];

                        var toTarget = target - pos;
                        var distTarget = toTarget.Length();
                        var normTarget = distTarget > 0.001f ? toTarget / distTarget : Vector2.Zero;

                        var fromPred = pos - predator;
                        var distPred = fromPred.Length();
                        var fearStrength = distPred < PredatorFearRange
                            ? (PredatorFearRange - distPred) / PredatorFearRange
                            : 0f;
                        var fearX = distPred > 0.001f ? (fromPred.X / distPred) * fearStrength : 0f;
                        var fearZ = distPred > 0.001f ? (fromPred.Y / distPred) * fearStrength : 0f;

                        var idx = i * InputSize;
                        inputsSpan[idx + 0] = normTarget.X;
                        inputsSpan[idx + 1] = normTarget.Y;
                        inputsSpan[idx + 2] = fearX;
                        inputsSpan[idx + 3] = fearZ;
                    });
                }
            }
        }

        private static void InferAllBots(ReadOnlySpan<float> population, ReadOnlySpan<float> inputs, Span<float> outputs)
        {
            unsafe
            {
                fixed (float* popPtr = population)
                fixed (float* inPtr = inputs)
                fixed (float* outPtr = outputs)
                {
                    var pPtr = (IntPtr)popPtr;
                    var iPtr = (IntPtr)inPtr;
                    var oPtr = (IntPtr)outPtr;
                    var popLen = population.Length;
                    var inLen = inputs.Length;
                    var outLen = outputs.Length;

                    Parallel.For(0, SwarmSize, i =>
                    {
                        var pop = new ReadOnlySpan<float>((float*)pPtr, popLen);
                        var ins = new ReadOnlySpan<float>((float*)iPtr, inLen);
                        var outs = new Span<float>((float*)oPtr, outLen);

                        var gIdx = i * GenomeSize;
                        var iIdx = i * InputSize;
                        var oIdx = i * OutputSize;

                        var x0 = ins[iIdx + 0];
                        var x1 = ins[iIdx + 1];
                        var x2 = ins[iIdx + 2];
                        var x3 = ins[iIdx + 3];

                        var y0 = x0 * pop[gIdx + 0]
                               + x1 * pop[gIdx + 1]
                               + x2 * pop[gIdx + 2]
                               + x3 * pop[gIdx + 3]
                               + pop[gIdx + 8];

                        var y1 = x0 * pop[gIdx + 4]
                               + x1 * pop[gIdx + 5]
                               + x2 * pop[gIdx + 6]
                               + x3 * pop[gIdx + 7]
                               + pop[gIdx + 9];

                        outs[oIdx + 0] = MathF.Tanh(y0);
                        outs[oIdx + 1] = MathF.Tanh(y1);
                    });
                }
            }
        }

        /// <summary>
        /// Training physics: respawn on death/out-of-bounds with fitness penalty.
        /// Fitness peaks at OrbitRadius band around target.
        /// </summary>
        private static void StepPhysicsAndFitness(
            Vector2[] positions, Vector2[] velocities, ReadOnlySpan<float> outputs,
            Vector2 target, Vector2 predator, float[] fitness, Random rng)
        {
            unsafe
            {
                fixed (float* outPtr = outputs)
                {
                    var oPtr = (IntPtr)outPtr;
                    var outLen = outputs.Length;

                    Parallel.For(0, SwarmSize,
                        () => new Random(Guid.NewGuid().GetHashCode()),
                        (i, state, localRng) =>
                        {
                            var outs = new ReadOnlySpan<float>((float*)oPtr, outLen);

                            var accel = new Vector2(outs[i * 2 + 0], outs[i * 2 + 1]);
                            velocities[i] = (velocities[i] + accel * (DeltaTime * AccelerationScale)) * Damping;
                            positions[i] += velocities[i] * DeltaTime;

                            var distTarget = Vector2.Distance(positions[i], target);
                            var distPred = Vector2.Distance(positions[i], predator);

                            // PREDATOR DEATH - soft penalty (not multiplicative - don't wipe out learned behavior)
                            if (distPred < PredatorRadius)
                            {
                                fitness[i] -= 20f;  // Fixed cost, not proportional
                                RespawnBot(ref positions[i], ref velocities[i], target, localRng);
                                return localRng;
                            }

                            // OUT OF ARENA
                            if (MathF.Abs(positions[i].X) > ArenaSize || MathF.Abs(positions[i].Y) > ArenaSize)
                            {
                                fitness[i] -= 10f;
                                RespawnBot(ref positions[i], ref velocities[i], target, localRng);
                                return localRng;
                            }

                            // PRIMARY REWARD: be at OrbitRadius distance from target
                            // Simplified - no tangent bonus (was causing spinning-in-place exploit)
                            var orbitDeviation = MathF.Abs(distTarget - OrbitRadius);
                            var orbitReward = 2f / (1f + orbitDeviation);  // 2.0 at perfect orbit, 1.0 at 1m off

                            // SECONDARY: tangential motion bonus only if near orbit radius
                            if (orbitDeviation < OrbitTolerance * 2f)
                            {
                                var fromTarget = positions[i] - target;
                                if (fromTarget.LengthSquared() > 0.01f)
                                {
                                    var radial = Vector2.Normalize(fromTarget);
                                    var tangent = new Vector2(-radial.Y, radial.X);
                                    var tangentialSpeed = MathF.Abs(Vector2.Dot(velocities[i], tangent));
                                    orbitReward += 0.2f * MathF.Min(tangentialSpeed, 2f);
                                }
                            }

                            fitness[i] += orbitReward;
                            return localRng;
                        },
                        _ => { });
                }
            }
        }

        private static void StepPhysicsDemo(
            Vector2[] positions, Vector2[] velocities, ReadOnlySpan<float> outputs,
            Vector2 target, Vector2 predator, Random rng)
        {
            unsafe
            {
                fixed (float* outPtr = outputs)
                {
                    var oPtr = (IntPtr)outPtr;
                    var outLen = outputs.Length;

                    Parallel.For(0, SwarmSize,
                        () => new Random(Guid.NewGuid().GetHashCode()),
                        (i, state, localRng) =>
                        {
                            var outs = new ReadOnlySpan<float>((float*)oPtr, outLen);

                            var accel = new Vector2(outs[i * 2 + 0], outs[i * 2 + 1]);
                            velocities[i] = (velocities[i] + accel * (DeltaTime * AccelerationScale)) * Damping;
                            positions[i] += velocities[i] * DeltaTime;

                            var distPred = Vector2.Distance(positions[i], predator);
                            var outOfBounds = MathF.Abs(positions[i].X) > ArenaSize || MathF.Abs(positions[i].Y) > ArenaSize;

                            if (distPred < PredatorRadius || outOfBounds)
                            {
                                RespawnBot(ref positions[i], ref velocities[i], target, localRng);
                            }

                            return localRng;
                        },
                        _ => { });
                }
            }
        }

        // ================================================================
        // GENETIC ALGORITHM
        // ================================================================

        private static (float best, float avg) DoGenerationStep(Span<float> population, float[] fitness, int[] indices, Random rng)
        {
            Array.Sort(indices, (a, b) => fitness[b].CompareTo(fitness[a]));

            var eliteCount = (int)(SwarmSize * ElitePercent);
            var best = fitness[indices[0]];

            var sum = 0f;
            for (var i = 0; i < SwarmSize; i++)
            {
                sum += fitness[i];
            }
            var avg = sum / SwarmSize;

            // Replace bottom (100 - elite%) with mutated offspring of elite
            for (var i = eliteCount; i < SwarmSize; i++)
            {
                var weakIdx = indices[i];
                var parentIdx = indices[rng.Next(0, eliteCount)];

                var weakBase = weakIdx * GenomeSize;
                var parentBase = parentIdx * GenomeSize;

                for (var g = 0; g < GenomeSize; g++)
                {
                    var mutation = rng.NextDouble() < MutationRate
                        ? (float)(rng.NextDouble() * 2 - 1) * MutationLarge
                        : (float)(rng.NextDouble() * 2 - 1) * MutationSmall;

                    var gene = population[parentBase + g] + mutation;
                    population[weakBase + g] = Math.Clamp(gene, -WeightClamp, WeightClamp);
                }
            }

            return (best, avg);
        }

        private static void InitPopulation(Span<float> population, Random rng)
        {
            for (var i = 0; i < population.Length; i++)
            {
                population[i] = (float)(rng.NextDouble() * 0.6 - 0.3);
            }
        }

        // ================================================================
        // PHYSICS HELPERS
        // ================================================================

        private static void SpawnAll(Vector2[] positions, Vector2[] velocities, Vector2 center, Random rng)
        {
            for (var i = 0; i < SwarmSize; i++)
            {
                var angle = (float)(rng.NextDouble() * Math.PI * 2);
                var radius = OrbitRadius + (float)(rng.NextDouble() - 0.5) * 4f;
                positions[i] = center + new Vector2(MathF.Cos(angle) * radius, MathF.Sin(angle) * radius);
                velocities[i] = Vector2.Zero;
            }
        }

        private static void RespawnBot(ref Vector2 position, ref Vector2 velocity, Vector2 target, Random rng)
        {
            // Respawn on ring around target, but clamped to be inside arena.
            // If target is near arena edge, spawn closer to prevent instant re-respawn loop.
            var maxSpawnRadius = ArenaSize - MathF.Max(MathF.Abs(target.X), MathF.Abs(target.Y)) - 1f;
            var spawnRadius = MathF.Min(OrbitRadius * 2f, MathF.Max(maxSpawnRadius, 2f));

            var angle = (float)(rng.NextDouble() * Math.PI * 2);
            position = target + new Vector2(MathF.Cos(angle) * spawnRadius, MathF.Sin(angle) * spawnRadius);
            velocity = Vector2.Zero;
        }

        // ================================================================
        // SERIALIZATION
        // ================================================================

        private static float[]? LoadBrain()
        {
            if (!File.Exists(BrainPath))
            {
                return null;
            }

            var bytes = File.ReadAllBytes(BrainPath);
            if (bytes.Length != GenomeSize * sizeof(float))
            {
                Console.Error.WriteLine($"[WARN] Brain file size mismatch: {bytes.Length} vs expected {GenomeSize * 4}");
                return null;
            }

            var brain = new float[GenomeSize];
            Buffer.BlockCopy(bytes, 0, brain, 0, bytes.Length);

            foreach (var w in brain)
            {
                if (float.IsNaN(w) || float.IsInfinity(w))
                {
                    Console.Error.WriteLine("[WARN] Brain contains NaN/Inf. Rejecting.");
                    return null;
                }
            }

            return brain;
        }

        private static void SaveBrain(ReadOnlySpan<float> population, int bestIndex)
        {
            var brain = new float[GenomeSize];
            population.Slice(bestIndex * GenomeSize, GenomeSize).CopyTo(brain);
            var bytes = new byte[GenomeSize * sizeof(float)];
            Buffer.BlockCopy(brain, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(BrainPath, bytes);
        }

        private static void SavePopulation(FastTensor<float> population)
        {
            SavePopulationFromSpan(population.GetView().AsReadOnlySpan());
        }

        private static void SavePopulationFromSpan(ReadOnlySpan<float> population)
        {
            var bytes = new byte[population.Length * sizeof(float)];
            MemoryMarshal.AsBytes(population).CopyTo(bytes);
            File.WriteAllBytes(PopulationPath, bytes);
        }

        private static void TryLoadPopulation(FastTensor<float> population)
        {
            if (!File.Exists(PopulationPath))
            {
                return;
            }

            var bytes = File.ReadAllBytes(PopulationPath);
            var expected = SwarmSize * GenomeSize * sizeof(float);
            if (bytes.Length != expected)
            {
                Console.WriteLine($"[WARN] Population checkpoint size mismatch ({bytes.Length} vs {expected}). Starting fresh.");
                return;
            }

            var span = MemoryMarshal.AsBytes(population.GetView().AsSpan());
            bytes.CopyTo(span);
            Console.WriteLine($"[OK] Loaded population checkpoint (generation {LoadGeneration()}).");
        }

        private static int LoadGeneration()
        {
            var path = Path.ChangeExtension(PopulationPath, ".gen");
            return File.Exists(path) && int.TryParse(File.ReadAllText(path), out var gen) ? gen : 0;
        }

        private static void SaveGeneration(int generation)
        {
            File.WriteAllText(Path.ChangeExtension(PopulationPath, ".gen"), generation.ToString());
        }

        // ================================================================
        // NETWORK
        // ================================================================

        private static void ReadExactly(NetworkStream stream, byte[] buffer)
        {
            var total = 0;
            while (total < buffer.Length)
            {
                var read = stream.Read(buffer, total, buffer.Length - total);
                if (read == 0)
                {
                    throw new IOException("Stream closed mid-frame.");
                }
                total += read;
            }
        }

        private static void WritePositions(Vector2[] positions, byte[] buffer)
        {
            unsafe
            {
                fixed (byte* pBuf = buffer)
                {
                    var ptr = (IntPtr)pBuf;
                    Parallel.For(0, SwarmSize, i =>
                    {
                        var p = positions[i];
                        var floats = (float*)ptr;
                        floats[i * 2] = p.X;
                        floats[i * 2 + 1] = p.Y;
                    });
                }
            }
        }

        private static void SetupShutdown(CancellationTokenSource cts)
        {
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
                Console.WriteLine("\n[SHUTDOWN] Stopping...");
            };
        }
    }
}