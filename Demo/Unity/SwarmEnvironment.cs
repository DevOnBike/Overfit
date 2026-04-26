// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;

namespace DevOnBike.Overfit.Demo.Unity.Server
{
    /// <summary>
    ///     Vectorised 2-D physics simulation for the swarm demo. Owns bot positions and
    ///     velocities, steps them forward using per-bot accelerations produced by the neural
    ///     policy, and accumulates a scalar fitness score per genome (a cohort of bots sharing
    ///     the same policy).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Concept: the population of ES candidates is small (hundreds) relative to the
    ///         visible swarm (hundreds of thousands). Each genome drives a <em>cohort</em> of
    ///         <see cref="BotsPerGenome"/> bots, and the genome's fitness is the sum of rewards
    ///         collected by its cohort. That averaging turns per-bot luck (lucky respawn
    ///         positions, transient NaNs) into a stable signal for the search gradient.
    ///     </para>
    ///     <para>
    ///         All bots live in a single shared arena: the same target and predator for
    ///         everyone each frame. This preserves the visual intuition of a single swarm
    ///         while still producing per-genome fitness — two bots from different cohorts
    ///         simply have different policies reacting to the same inputs.
    ///     </para>
    ///     <para>
    ///         The object is thread-compatible but not thread-safe: internal storage is
    ///         reused frame to frame. Physics stepping internally uses
    ///         <see cref="Parallel.For"/>.
    ///     </para>
    /// </remarks>
    public sealed class SwarmEnvironment
    {
        private readonly SwarmConfig _config;
        private readonly Vector2[] _positions;
        private readonly Vector2[] _velocities;
        private readonly int[] _genomeOfBot;

        public SwarmEnvironment(SwarmConfig config)
        {
            ArgumentNullException.ThrowIfNull(config);

            if (config.SwarmSize % config.PopulationSize != 0)
            {
                throw new ArgumentException(
                    $"SwarmSize ({config.SwarmSize}) must be divisible by PopulationSize " +
                    $"({config.PopulationSize}) so cohorts are uniform.",
                    nameof(config));
            }

            _config = config;
            _positions = new Vector2[config.SwarmSize];
            _velocities = new Vector2[config.SwarmSize];

            BotsPerGenome = config.SwarmSize / config.PopulationSize;

            // Pre-compute the genome→bot mapping once. For bot index i,
            // _genomeOfBot[i] tells the physics step which policy owns that bot.
            _genomeOfBot = new int[config.SwarmSize];
            for (var i = 0; i < config.SwarmSize; i++)
            {
                _genomeOfBot[i] = i / BotsPerGenome;
            }
        }

        /// <summary>
        ///     How many bots each genome controls. Bots are assigned to genomes in contiguous
        ///     blocks: bots [0..BotsPerGenome) → genome 0, next block → genome 1, and so on.
        ///     Kept constant across the life of the environment.
        /// </summary>
        public int BotsPerGenome { get; }

        public ReadOnlySpan<Vector2> Positions => _positions;

        /// <summary>
        ///     Resets every bot to a random ring around <paramref name="center"/>. Called at
        ///     the start of every generation so no cohort inherits positional advantages
        ///     from the previous generation's trajectory.
        /// </summary>
        public void Respawn(Vector2 center, Random rng)
        {
            for (var i = 0; i < _config.SwarmSize; i++)
            {
                RespawnOne(i, center, rng);
            }
        }

        /// <summary>
        ///     Builds the 4-float input vector for every bot into <paramref name="inputs"/>,
        ///     laid out as <c>[swarmSize × inputSize]</c>. Inputs are the bot's normalised
        ///     direction to target and a range-bounded fear vector away from the predator.
        /// </summary>
        public void BuildInputs(Vector2 target, Vector2 predator, Span<float> inputs)
        {
            if (inputs.Length != _config.SwarmSize * _config.InputSize)
            {
                throw new ArgumentException(
                    $"inputs must have length SwarmSize * InputSize = {_config.SwarmSize * _config.InputSize}.",
                    nameof(inputs));
            }

            var swarmSize = _config.SwarmSize;
            var inputSize = _config.InputSize;
            var fearRange = _config.PredatorFearRange;

            // Spans cannot be captured in lambdas, so we hand-roll the unsafe pointer trick
            // here exactly as the older implementation did — this is the one place in the
            // demo that earns its unsafe block.
            unsafe
            {
                fixed (float* inputsPtr = inputs)
                {
                    var ptr = (IntPtr)inputsPtr;
                    var positions = _positions;

                    Parallel.For(0, swarmSize, i =>
                    {
                        var span = new Span<float>((float*)ptr, swarmSize * inputSize);
                        var pos = positions[i];

                        var toTarget = target - pos;
                        var distTarget = toTarget.Length();
                        var normTarget = distTarget > 0.001f ? toTarget / distTarget : Vector2.Zero;

                        var fromPred = pos - predator;
                        var distPred = fromPred.Length();
                        var fearStrength = distPred < fearRange ? (fearRange - distPred) / fearRange : 0f;
                        var fearX = distPred > 0.001f ? (fromPred.X / distPred) * fearStrength : 0f;
                        var fearY = distPred > 0.001f ? (fromPred.Y / distPred) * fearStrength : 0f;

                        var baseIdx = i * inputSize;
                        span[baseIdx + 0] = normTarget.X;
                        span[baseIdx + 1] = normTarget.Y;
                        span[baseIdx + 2] = fearX;
                        span[baseIdx + 3] = fearY;
                    });
                }
            }
        }

        /// <summary>
        ///     Advances the simulation by one frame using per-bot accelerations in
        ///     <paramref name="outputs"/> (laid out as <c>[swarmSize × outputSize]</c>),
        ///     accumulating per-genome fitness into <paramref name="genomeFitness"/>.
        ///     Bots that die or leave the arena are respawned at a ring around
        ///     <paramref name="target"/>.
        /// </summary>
        /// <param name="genomeFitness">
        ///     Sum-accumulated per-genome fitness. Length must equal
        ///     <see cref="SwarmConfig.PopulationSize"/>. The environment adds to existing
        ///     values — clear before each generation.
        /// </param>
        public void Step(
            ReadOnlySpan<float> outputs,
            Vector2 target,
            Vector2 predator,
            Span<float> genomeFitness,
            Random rng)
        {
            if (outputs.Length != _config.SwarmSize * _config.OutputSize)
            {
                throw new ArgumentException(
                    $"outputs must have length SwarmSize * OutputSize = {_config.SwarmSize * _config.OutputSize}.",
                    nameof(outputs));
            }

            if (genomeFitness.Length != _config.PopulationSize)
            {
                throw new ArgumentException(
                    $"genomeFitness must have length PopulationSize = {_config.PopulationSize}.",
                    nameof(genomeFitness));
            }

            // Fitness accumulation must be parallel-safe. We reserve one slot per genome
            // and use Interlocked-adds across threads. Given population ≤ ~1024 and bots in
            // the 100k range, contention is negligible in practice.
            unsafe
            {
                fixed (float* outPtr = outputs)
                fixed (float* fitPtr = genomeFitness)
                {
                    var oPtr = (IntPtr)outPtr;
                    var fPtr = (IntPtr)fitPtr;
                    var outLen = outputs.Length;
                    var fitLen = genomeFitness.Length;

                    Parallel.For(0, _config.SwarmSize,
                        () => new Random(Guid.NewGuid().GetHashCode()),
                        (i, _, localRng) =>
                        {
                            StepOne(i, oPtr, outLen, fPtr, fitLen, target, predator, localRng);
                            return localRng;
                        },
                        _ => { });
                }
            }
        }

        /// <summary>
        ///     Simpler step routine for the demo mode: physics only, no fitness accumulation
        ///     and no per-genome rewards. Dead / out-of-arena bots still respawn so the Unity
        ///     client sees a full arena at all times.
        /// </summary>
        public void StepDemo(ReadOnlySpan<float> outputs, Vector2 target, Vector2 predator, Random rng)
        {
            if (outputs.Length != _config.SwarmSize * _config.OutputSize)
            {
                throw new ArgumentException(
                    $"outputs must have length SwarmSize * OutputSize = {_config.SwarmSize * _config.OutputSize}.",
                    nameof(outputs));
            }

            unsafe
            {
                fixed (float* outPtr = outputs)
                {
                    var oPtr = (IntPtr)outPtr;
                    var outLen = outputs.Length;

                    Parallel.For(0, _config.SwarmSize,
                        () => new Random(Guid.NewGuid().GetHashCode()),
                        (i, _, localRng) =>
                        {
                            StepDemoOne(i, oPtr, outLen, target, predator, localRng);
                            return localRng;
                        },
                        _ => { });
                }
            }
        }

        // -------------------------------------------------------------------------
        // Inner loop — training step
        // -------------------------------------------------------------------------

        private unsafe void StepOne(
            int i,
            IntPtr outPtrRaw, int outLen,
            IntPtr fitPtrRaw, int fitLen,
            Vector2 target, Vector2 predator,
            Random localRng)
        {
            var outs = new ReadOnlySpan<float>((float*)outPtrRaw, outLen);
            var fit = new Span<float>((float*)fitPtrRaw, fitLen);

            var outBase = i * _config.OutputSize;
            var accel = new Vector2(outs[outBase + 0], outs[outBase + 1]);

            _velocities[i] = (_velocities[i] + (accel * (_config.DeltaTime * _config.AccelerationScale))) * _config.Damping;
            _positions[i] += _velocities[i] * _config.DeltaTime;

            var distTarget = Vector2.Distance(_positions[i], target);
            var distPred = Vector2.Distance(_positions[i], predator);
            var genome = _genomeOfBot[i];

            if (distPred < _config.PredatorRadius)
            {
                AddFitness(fit, genome, -_config.PredatorPenalty);
                RespawnOne(i, target, localRng);
                return;
            }

            if (MathF.Abs(_positions[i].X) > _config.ArenaSize ||
                MathF.Abs(_positions[i].Y) > _config.ArenaSize)
            {
                AddFitness(fit, genome, -_config.ArenaPenalty);
                RespawnOne(i, target, localRng);
                return;
            }

            // Primary reward: stay at OrbitRadius from the target. The 1/(1+ε) shape gives a
            // smooth gradient all the way from far-out to perfectly on-ring.
            var orbitDeviation = MathF.Abs(distTarget - _config.OrbitRadius);
            var orbitReward = _config.OrbitRewardScale / (1f + orbitDeviation);

            // Secondary: tangential-motion bonus — rewards actually orbiting rather than
            // sitting still on the ring. Only active near the ring so it doesn't reward
            // drifting sideways far from the target.
            if (orbitDeviation < _config.OrbitTolerance * 2f)
            {
                var fromTarget = _positions[i] - target;

                if (fromTarget.LengthSquared() > 0.01f)
                {
                    var radial = Vector2.Normalize(fromTarget);
                    var tangent = new Vector2(-radial.Y, radial.X);
                    var tangentialSpeed = MathF.Abs(Vector2.Dot(_velocities[i], tangent));
                    orbitReward += _config.TangentBonusScale * MathF.Min(tangentialSpeed, 2f);
                }
            }

            AddFitness(fit, genome, orbitReward);
        }

        // -------------------------------------------------------------------------
        // Inner loop — demo step (no fitness)
        // -------------------------------------------------------------------------

        private unsafe void StepDemoOne(
            int i,
            IntPtr outPtrRaw, int outLen,
            Vector2 target, Vector2 predator,
            Random localRng)
        {
            var outs = new ReadOnlySpan<float>((float*)outPtrRaw, outLen);

            var outBase = i * _config.OutputSize;
            var accel = new Vector2(outs[outBase + 0], outs[outBase + 1]);

            _velocities[i] = (_velocities[i] + (accel * (_config.DeltaTime * _config.AccelerationScale))) * _config.Damping;
            _positions[i] += _velocities[i] * _config.DeltaTime;

            var distPred = Vector2.Distance(_positions[i], predator);
            var outOfBounds =
                MathF.Abs(_positions[i].X) > _config.ArenaSize ||
                MathF.Abs(_positions[i].Y) > _config.ArenaSize;

            if (distPred < _config.PredatorRadius || outOfBounds)
            {
                RespawnOne(i, target, localRng);
            }
        }

        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private static void AddFitness(Span<float> fitness, int genomeIndex, float delta)
        {
            // Interlocked.Add has no float overload but CompareExchange does. This is a
            // standard lock-free float-accumulator idiom.
            ref var slot = ref fitness[genomeIndex];

            float current;
            float updated;

            do
            {
                current = slot;
                updated = current + delta;
            }
            while (Interlocked.CompareExchange(ref slot, updated, current) != current);
        }

        private void RespawnOne(int i, Vector2 center, Random rng)
        {
            // Respawn on a ring around the target, but clamp radius so the spawn point
            // stays inside the arena — without this, targets near the arena edge cause
            // instant-respawn loops (the bot spawns out of bounds and dies on the next
            // frame).
            var maxSpawnRadius = _config.ArenaSize - MathF.Max(MathF.Abs(center.X), MathF.Abs(center.Y)) - 1f;
            var spawnRadius = MathF.Min(_config.OrbitRadius * 2f, MathF.Max(maxSpawnRadius, 2f));

            var angle = (float)(rng.NextDouble() * Math.PI * 2);
            _positions[i] = center + new Vector2(MathF.Cos(angle) * spawnRadius, MathF.Sin(angle) * spawnRadius);
            _velocities[i] = Vector2.Zero;
        }
    }
}
