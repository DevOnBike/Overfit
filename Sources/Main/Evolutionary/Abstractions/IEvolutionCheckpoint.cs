// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    /// <summary>
    ///     Persists an evolutionary algorithm's in-memory state to a binary stream and
    ///     restores it later. Designed for long-running training jobs that must survive
    ///     process restarts.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Implementations write a self-describing header (magic number + schema version
    ///         + dimensional invariants) so <see cref="Load"/> can reject streams produced by
    ///         a different algorithm or a mismatched configuration.
    ///     </para>
    ///     <para>
    ///         Checkpoints capture the observable state of the algorithm: population or mean
    ///         vector, generation counter, best-candidate tracking. They do NOT capture the
    ///         internal state of the <see cref="System.Random"/> instance — after
    ///         <see cref="Load"/> the RNG restarts from a fresh seed. Evolutionary training
    ///         is inherently stochastic, so this is a deliberate simplification rather than a
    ///         correctness issue: a resumed run is statistically equivalent to continuing the
    ///         original, but not bit-identical.
    ///     </para>
    ///     <para>
    ///         Checkpoints should only be written between complete Ask/Tell cycles. Saving
    ///         mid-generation (after Ask but before Tell) produces a stream that Load cannot
    ///         interpret as a valid post-generation state.
    ///     </para>
    /// </remarks>
    public interface IEvolutionCheckpoint
    {
        /// <summary>
        ///     Writes the algorithm's state to the given <see cref="BinaryWriter"/>.
        ///     The stream should be positioned where the implementation wants to begin writing
        ///     and will be advanced past the final byte on return.
        /// </summary>
        void Save(BinaryWriter writer);

        /// <summary>
        ///     Reads algorithm state from the given <see cref="BinaryReader"/>, overwriting
        ///     the current in-memory state. Throws <see cref="InvalidDataException"/> if the
        ///     stream's header does not match this algorithm or its configuration (population
        ///     size, parameter count, algorithm identifier).
        /// </summary>
        void Load(BinaryReader reader);
    }
}