// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Evolutionary.Storage
{
    /// <summary>
    ///     Outcome of attempting to insert a candidate into an <see cref="IEliteArchive"/>.
    /// </summary>
    /// <remarks>
    ///     <see cref="InvalidFitness"/> is distinct from <see cref="Rejected"/> on purpose:
    ///     a rejection means the archive's quality criterion judged the candidate worse than
    ///     the existing elite, which is normal and expected during exploration. An invalid
    ///     fitness — NaN or ±∞ — is a contract violation by the evaluator and should be
    ///     surfaced as a separate signal so callers can detect and fix their fitness function
    ///     instead of silently dropping samples.
    /// </remarks>
    public enum EliteInsertStatus
    {
        Rejected = 0,
        InsertedNewCell = 1,
        ReplacedExistingCell = 2,
        OutOfBounds = 3,
        InvalidFitness = 4
    }
}