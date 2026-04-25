// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    /// <summary>
    ///     A shared buffer of pre-computed standard-normal samples used by
    ///     noise-table-style evolution strategies (e.g. OpenAI ES). Implementations must be
    ///     safe for concurrent read access once construction completes: multiple
    ///     <see cref="GetSlice"/> and <see cref="SampleOffset"/> calls may run in parallel
    ///     from fitness-evaluation worker threads.
    /// </summary>
    public interface INoiseTable
    {
        /// <summary>
        ///     Total number of samples in the table.
        /// </summary>
        int Length { get; }

        /// <summary>
        ///     Returns a zero-allocation view over <paramref name="length"/> consecutive
        ///     samples starting at <paramref name="offset"/>. The returned span must remain
        ///     valid for the caller's lifetime of the slice (implementations must not mutate
        ///     or reallocate the underlying storage during read).
        /// </summary>
        ReadOnlySpan<float> GetSlice(int offset, int length);

        /// <summary>
        ///     Uniformly samples an offset within <c>[0, Length − sliceLength]</c>, i.e. any
        ///     offset at which a slice of <paramref name="sliceLength"/> elements fits
        ///     entirely inside the table. Every returned offset must be a legal argument to
        ///     <see cref="GetSlice"/> paired with the same <paramref name="sliceLength"/>.
        /// </summary>
        int SampleOffset(Random rng, int sliceLength);
    }
}