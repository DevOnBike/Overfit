using System.Collections.Concurrent;
using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Randomization;

namespace DevOnBike.Overfit.Evolutionary.Storage
{
    /// <summary>
    ///     A pre-computed block of standard-normal samples backing
    ///     <see cref="INoiseTable"/>. Designed for Salimans-style Evolution Strategies,
    ///     where many perturbation vectors are drawn each generation from a shared pool.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The table is filled once in the constructor via parallel Box-Muller sampling.
    ///         Each partition runs an independent <see cref="VectorizedRandom"/> seeded
    ///         deterministically from the caller-supplied master seed, so rebuilding with
    ///         the same seed produces the same bytes even though the fill runs across
    ///         threads with non-deterministic scheduling. After construction the instance is
    ///         immutable: <see cref="GetSlice"/> returns zero-allocation
    ///         <see cref="ReadOnlySpan{T}"/> views and <see cref="SampleOffset"/> performs a
    ///         single <see cref="Random.Next(int)"/> call.
    ///     </para>
    ///     <para>
    ///         The algorithmic advantage of a noise table is that each perturbation is
    ///         identified by a single integer offset rather than a full parameter-sized
    ///         vector. This is the mechanism OpenAI ES uses to scale to distributed workers:
    ///         only the offset travels across the network, and every worker reconstructs
    ///         the same noise vector from its local copy of the table.
    ///     </para>
    ///     <para>
    ///         Memory footprint is exactly <c>length * sizeof(float)</c> bytes. A typical
    ///         reinforcement-learning agent uses a 16M-float table (~64 MB), which amortizes
    ///         well across long training runs but is substantially smaller than Salimans' original
    ///         250 MB setting. Increase the length for very large genomes (>100k parameters)
    ///         to reduce correlation between perturbations.
    ///     </para>
    ///     <para>
    ///         <b>Breaking behavioural change (round 14):</b> the per-partition RNG was
    ///         <see cref="Random"/> in earlier versions and is now <see cref="VectorizedRandom"/>.
    ///         The two RNGs produce different bit streams from the same seed, so a table built
    ///         with the same master seed as before will contain different (but statistically
    ///         equivalent) noise samples. Downstream code that snapshots specific float values
    ///         from the table needs to be re-baselined.
    ///     </para>
    /// </remarks>
    public sealed class PrecomputedNoiseTable : INoiseTable
    {
        private const int MinPartitionSize = 4096;

        private readonly float[] _buffer;

        public PrecomputedNoiseTable(int length, int? seed = null)
        {
            if (length <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(length), "Noise-table length must be positive.");
            }

            _buffer = new float[length];

            // A single master seed fans out into deterministic per-partition seeds so that
            // rebuilding the table with the same seed produces the same bytes, even though
            // the fill itself runs in parallel with non-deterministic scheduling.
            var masterSeed = seed ?? new Random().Next();

            var partitionSize = Math.Max(MinPartitionSize, length / Math.Max(1, Environment.ProcessorCount));
            var partitioner = Partitioner.Create(0, length, partitionSize);

            Parallel.ForEach(partitioner, range =>
            {
                var (from, to) = range;

                // VectorizedRandom's constructor takes a uint; we derive the per-partition
                // seed from (masterSeed, from) so independent partitions decorrelate. A cast
                // from int to uint preserves all bit patterns, including negative values.
                var partitionSeed = unchecked((uint)HashCombine(masterSeed, from));
                var partitionRng = new VectorizedRandom(partitionSeed);

                FillRange(_buffer.AsSpan(from, to - from), partitionRng);
            });
        }

        public int Length => _buffer.Length;

        public ReadOnlySpan<float> GetSlice(int offset, int length)
        {
            if ((uint)offset > (uint)_buffer.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(offset));
            }

            if (length < 0 || offset + length > _buffer.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(length));
            }

            return new ReadOnlySpan<float>(_buffer, offset, length);
        }

        /// <summary>
        ///     Uniformly samples an offset in <c>[0, Length - sliceLength]</c> — i.e. any
        ///     offset at which a slice of <paramref name="sliceLength"/> elements fits
        ///     entirely inside the table. The returned offset is safe to feed to
        ///     <see cref="GetSlice"/> with the same <paramref name="sliceLength"/>.
        /// </summary>
        public int SampleOffset(Random rng, int sliceLength)
        {
            ArgumentNullException.ThrowIfNull(rng);

            if (sliceLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(sliceLength));
            }

            if (sliceLength > _buffer.Length)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(sliceLength),
                    $"sliceLength {sliceLength} exceeds table length {_buffer.Length}.");
            }

            // Random.Next(exclusiveMax) returns [0, exclusiveMax). We need [0, length - slice],
            // which is an inclusive upper bound, hence the +1.
            var exclusiveUpper = _buffer.Length - sliceLength + 1;
            return rng.Next(exclusiveUpper);
        }

        private static void FillRange(Span<float> target, VectorizedRandom rng)
        {
            // Paired Box-Muller: every pair of uniform draws produces two independent
            // standard-normal samples, written to adjacent positions. This halves the
            // per-element Log/Sqrt/SinCos cost compared with discarding-sin Box-Muller.
            //
            // VectorizedRandom.NextSingle() reads from a lane cache backed by 256-bit SIMD
            // steps — eight uniforms are produced at once and consumed scalar-by-scalar.
            // The Box-Muller transcendentals (Log, SinCos) remain scalar; they dominate
            // runtime at the ~80% mark, so the end-to-end speedup is closer to 20–40%
            // rather than 8×. Still a measurable win, and the constructor only runs once
            // per strategy initialisation so it is not on the generation hot path.
            var i = 0;

            while (i + 1 < target.Length)
            {
                float u1;

                do
                {
                    u1 = rng.NextSingle();
                }
                while (u1 == 0f);

                var u2 = rng.NextSingle();

                var mag = MathF.Sqrt(-2f * MathF.Log(u1));
                var (sin, cos) = MathF.SinCos(2f * MathF.PI * u2);

                target[i] = mag * cos;
                target[i + 1] = mag * sin;
                i += 2;
            }

            // Odd leftover: fall back to single-value Box-Muller.
            if (i < target.Length)
            {
                float u1;

                do
                {
                    u1 = rng.NextSingle();
                }
                while (u1 == 0f);

                var u2 = rng.NextSingle();
                target[i] = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
            }
        }

        private static int HashCombine(int a, int b)
        {
            // Boost-style combine: good enough to decorrelate adjacent partition seeds.
            var h = (uint)a;
            h ^= (uint)b + 0x9E3779B9u + (h << 6) + (h >> 2);
            return (int)h;
        }
    }
}