// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Numerics;
using DevOnBike.Overfit.Evolutionary.Fitness;
using DevOnBike.Overfit.Evolutionary.Storage;
using DevOnBike.Overfit.Evolutionary.Strategies;

namespace DevOnBike.Overfit.Demo.Unity.Server
{
    /// <summary>
    ///     Orchestrates training: maintains one <see cref="SwarmBrain"/> per ES candidate,
    ///     drives the shared <see cref="SwarmEnvironment"/>, and pipes per-generation
    ///     genomes in and out of <see cref="OpenAiEsStrategy"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Per generation the trainer:
    ///         <list type="number">
    ///             <item>Calls <see cref="OpenAiEsStrategy.Ask"/> to get a flat matrix of
    ///                   candidate genomes (<c>population × parameterCount</c>).</item>
    ///             <item>Loads each genome into its dedicated <see cref="SwarmBrain"/>.</item>
    ///             <item>Runs the environment for <see cref="SwarmConfig.FramesPerGeneration"/>
    ///                   frames, accumulating per-genome fitness.</item>
    ///             <item>Passes fitness back via <see cref="OpenAiEsStrategy.Tell"/>, which
    ///                   updates μ using the centered-rank-shaped gradient + Adam.</item>
    ///         </list>
    ///     </para>
    ///     <para>
    ///         The checkpoint API of <see cref="OpenAiEsStrategy"/> is wired up so
    ///         long-running trainings can be stopped with Ctrl+C and resumed on the next run:
    ///         whenever <paramref name="checkpointPath"/> exists on start, it is loaded; the
    ///         final μ is saved on clean shutdown. Best brain seen so far is additionally
    ///         written to <paramref name="brainPath"/> every generation for the Unity client.
    ///     </para>
    /// </remarks>
    public sealed class SwarmTrainer : IDisposable
    {
        private readonly SwarmConfig _config;
        private readonly string _brainPath;
        private readonly string _checkpointPath;

        private readonly OpenAiEsStrategy _strategy;
        private readonly PrecomputedNoiseTable _noiseTable;
        private readonly SwarmEnvironment _environment;
        private readonly SwarmBrain[] _brains;
        private readonly SwarmBrain _bestBrain;

        private readonly float[] _genomes;          // [population × parameterCount]
        private readonly float[] _fitness;          // [population]
        private readonly float[] _inputBuffer;      // [swarmSize × inputSize]
        private readonly float[] _outputBuffer;     // [swarmSize × outputSize]

        public SwarmTrainer(SwarmConfig config, string brainPath, string checkpointPath)
        {
            ArgumentNullException.ThrowIfNull(config);
            ArgumentNullException.ThrowIfNull(brainPath);
            ArgumentNullException.ThrowIfNull(checkpointPath);

            _config = config;
            _brainPath = brainPath;
            _checkpointPath = checkpointPath;

            _noiseTable = new PrecomputedNoiseTable(config.NoiseTableLength, config.Seed);

            _strategy = new OpenAiEsStrategy(
                populationSize: config.PopulationSize,
                parameterCount: config.GenomeSize,
                sigma: config.Sigma,
                learningRate: config.LearningRate,
                noiseTable: _noiseTable,
                shaper: new CenteredRankFitnessShaper(),
                seed: config.Seed);

            _environment = new SwarmEnvironment(config);

            // One brain per ES candidate; they run in parallel from Parallel.For and each
            // invokes Sequential.ForwardInference, whose internal PooledBuffer<float> uses
            // a shared ArrayPool — safe under concurrency, cheap to reacquire.
            _brains = new SwarmBrain[config.PopulationSize];
            for (var i = 0; i < _brains.Length; i++)
            {
                _brains[i] = new SwarmBrain(config.InputSize, config.HiddenSize, config.OutputSize);
            }

            // Separate brain used to save the best-so-far genome to disk each generation.
            _bestBrain = new SwarmBrain(config.InputSize, config.HiddenSize, config.OutputSize);

            _genomes = new float[config.PopulationSize * config.GenomeSize];
            _fitness = new float[config.PopulationSize];
            _inputBuffer = new float[config.SwarmSize * config.InputSize];
            _outputBuffer = new float[config.SwarmSize * config.OutputSize];
        }

        /// <summary>
        ///     Runs a training loop for <paramref name="generations"/> generations, or until
        ///     <paramref name="cancellation"/> fires. Returns the best fitness ever observed.
        /// </summary>
        public float Train(int generations, CancellationToken cancellation)
        {
            if (generations <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(generations));
            }

            ResumeOrInitialize();

            var rng = _config.Seed is int s ? new Random(s) : new Random();
            var startGeneration = _strategy.Generation;
            var totalSw = Stopwatch.StartNew();

            for (var gen = startGeneration;
                 gen < startGeneration + generations && !cancellation.IsCancellationRequested;
                 gen++)
            {
                var genSw = Stopwatch.StartNew();

                RunGeneration(rng);

                var genTime = genSw.Elapsed.TotalSeconds;
                var eta = (startGeneration + generations - (gen + 1)) * genTime;

                // Per-generation statistics across the population, to make it visible whether
                // the search is actually moving. Useful when best_ever looks stuck: if gen_best
                // is trending upwards and gen_mean is far below gen_best, ES is exploring but
                // not yet improving the high-water mark. If gen_mean == gen_best == gen_worst,
                // the population has collapsed.
                ComputeGenerationStats(out var genBest, out var genMean, out var genWorst);

                Console.WriteLine(
                    $"[GEN {gen + 1,4}/{startGeneration + generations}] " +
                    $"best_ever={_strategy.BestFitness,10:F1}  " +
                    $"gen_best={genBest,10:F1}  " +
                    $"gen_mean={genMean,10:F1}  " +
                    $"gen_worst={genWorst,10:F1}  " +
                    $"t={genTime:F1}s");

                // Persist every generation — the serialisation is cheap (456 B) and the
                // expected-run latency between checkpoints dominates any IO cost. Also saves
                // the user from re-running when something goes wrong mid-training.
                SaveBestBrain();

                // Full state checkpoint every 10 generations so Ctrl+C loses at most 10 gens.
                if ((gen + 1) % 10 == 0)
                {
                    SaveCheckpoint();
                }
            }

            SaveCheckpoint();
            SaveBestBrain();

            Console.WriteLine($"[DONE] Total time: {totalSw.Elapsed.TotalMinutes:F1} min");
            Console.WriteLine($"[DONE] Best fitness: {_strategy.BestFitness:F1}");
            Console.WriteLine($"[DONE] Best brain: {_brainPath}");

            return _strategy.BestFitness;
        }

        // -------------------------------------------------------------------
        // Per-generation
        // -------------------------------------------------------------------

        private void RunGeneration(Random rng)
        {
            // Ask: sample this generation's population around μ.
            _strategy.Ask(_genomes);

            // Load each candidate's weights into its dedicated brain. Parallel across
            // candidates — LoadGenome does no shared-state work, the adapter writes to its
            // own network's tensors.
            Parallel.For(0, _config.PopulationSize, i =>
            {
                var slice = new ReadOnlySpan<float>(_genomes, i * _config.GenomeSize, _config.GenomeSize);
                _brains[i].LoadGenome(slice);
            });

            // Reset fitness accumulator and respawn all bots. Target starts at origin for
            // the first teleport; subsequent teleports happen every 150 frames.
            Array.Clear(_fitness);
            _environment.Respawn(Vector2.Zero, rng);

            var target = Vector2.Zero;
            var predator = new Vector2(3f, 3f);
            var predatorAngle = 0f;

            for (var frame = 0; frame < _config.FramesPerGeneration; frame++)
            {
                // Keep the problem non-trivial: teleport the target every 150 frames so the
                // policy has to recover its orbit from arbitrary starting positions.
                if (frame % 150 == 0)
                {
                    target = new Vector2(
                        (float)((rng.NextDouble() * 24.0) - 12.0),
                        (float)((rng.NextDouble() * 24.0) - 12.0));
                }

                predatorAngle += 0.005f;
                predator = new Vector2(MathF.Cos(predatorAngle) * 10f, MathF.Sin(predatorAngle) * 10f);

                _environment.BuildInputs(target, predator, _inputBuffer);

                RunInferenceForPopulation();

                _environment.Step(_outputBuffer, target, predator, _fitness, rng);
            }

            // Tell: pass fitness back to ES, which updates μ.
            _strategy.Tell(_fitness);
        }

        /// <summary>
        ///     Fan-out inference across all <see cref="SwarmConfig.PopulationSize"/> candidates.
        ///     Each brain handles its own <see cref="SwarmEnvironment.BotsPerGenome"/> bots
        ///     — contiguous in the input/output buffers.
        /// </summary>
        private void RunInferenceForPopulation()
        {
            var botsPerGenome = _environment.BotsPerGenome;
            var inputSize = _config.InputSize;
            var outputSize = _config.OutputSize;
            var population = _config.PopulationSize;

            Parallel.For(0, population, genomeIndex =>
            {
                var botStart = genomeIndex * botsPerGenome;
                var brain = _brains[genomeIndex];

                for (var b = 0; b < botsPerGenome; b++)
                {
                    var botIndex = botStart + b;

                    var input = new ReadOnlySpan<float>(
                        _inputBuffer, botIndex * inputSize, inputSize);
                    var output = new Span<float>(
                        _outputBuffer, botIndex * outputSize, outputSize);

                    brain.Infer(input, output);
                }
            });
        }

        // -------------------------------------------------------------------
        // Persistence
        // -------------------------------------------------------------------

        private void ResumeOrInitialize()
        {
            if (File.Exists(_checkpointPath))
            {
                try
                {
                    using var fs = File.OpenRead(_checkpointPath);
                    using var br = new BinaryReader(fs);
                    _strategy.Load(br);

                    Console.WriteLine(
                        $"[OK] Resumed training from generation {_strategy.Generation} " +
                        $"(best fitness so far: {_strategy.BestFitness:F1}).");
                    return;
                }
                catch (Exception ex) when (ex is InvalidDataException or IOException)
                {
                    Console.WriteLine($"[WARN] Could not load checkpoint ({ex.Message}); starting fresh.");
                }
            }

            _strategy.Initialize();
            Console.WriteLine("[OK] Fresh ES run initialised.");
        }

        private void SaveCheckpoint()
        {
            using var fs = File.Create(_checkpointPath);
            using var bw = new BinaryWriter(fs);
            _strategy.Save(bw);
        }

        private void SaveBestBrain()
        {
            // For ES, the policy you want to deploy is μ (the centre of the sampling
            // distribution), NOT the best candidate sampled so far. GetBestParameters()
            // returns a single lucky sample — it may hit a fitness outlier early and
            // then never be reproduced. μ represents the trajectory the gradient update
            // has followed, and every sample drawn from μ ± σε is a small perturbation
            // around it — the consistent policy.
            //
            // Concretely: on this swarm demo with a ceiling-like fitness (predator kills
            // and arena-out penalties cap the achievable reward), best-ever sits at the
            // first lucky sample and never moves. Mean improves steadily with the
            // gradient and produces the reproducible policy.
            var mean = _strategy.Mean;

            if (mean.IsEmpty)
            {
                return;
            }

            _bestBrain.LoadGenome(mean);
            _bestBrain.SaveToFile(_brainPath);
        }

        private void ComputeGenerationStats(out float best, out float mean, out float worst)
        {
            best = float.NegativeInfinity;
            worst = float.PositiveInfinity;
            var sum = 0.0;
            var count = 0;

            for (var i = 0; i < _fitness.Length; i++)
            {
                var f = _fitness[i];

                if (!float.IsFinite(f))
                {
                    continue;
                }

                if (f > best) { best = f; }
                if (f < worst) { worst = f; }
                sum += f;
                count++;
            }

            mean = count > 0 ? (float)(sum / count) : float.NaN;

            if (count == 0)
            {
                best = float.NaN;
                worst = float.NaN;
            }
        }

        public void Dispose()
        {
            foreach (var brain in _brains)
            {
                brain.Dispose();
            }

            _bestBrain.Dispose();
            _strategy.Dispose();
        }
    }
}