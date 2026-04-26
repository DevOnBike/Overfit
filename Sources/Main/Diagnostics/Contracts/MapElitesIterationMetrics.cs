// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Runtime;

namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    /// <summary>
    ///     Per-iteration summary emitted by <c>MapElites.RunIteration</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         <b>BestEvaluatedFitness vs BestEliteFitness.</b> These are deliberately split.
    ///         <c>BestEvaluatedFitness</c> is the strongest fitness ever produced by the
    ///         evaluator, regardless of whether the candidate ended up in the archive.
    ///         <c>BestEliteFitness</c> is the strongest fitness currently held by the archive
    ///         — i.e. the strongest *behavioural* elite that contributes to coverage.
    ///         For QD work the elite metric is usually the one to track; for sanity-checking
    ///         that the evaluator can actually find good candidates at all, evaluated is
    ///         the right one. They diverge whenever the strongest candidate was rejected
    ///         (out-of-bounds descriptor or beaten by an existing elite in the same cell).
    ///     </para>
    ///     <para>
    ///         <b>InvalidFitnessCount.</b> Counts candidates whose fitness was NaN or ±∞
    ///         and were therefore refused by the archive. Non-zero values indicate a bug
    ///         in the evaluator (e.g. division-by-zero in a reward function); they should
    ///         be alarming, not informational.
    ///     </para>
    /// </remarks>
    public readonly struct MapElitesIterationMetrics
    {
        public MapElitesIterationMetrics(
            int iteration,
            int insertedNewCells,
            int replacedExistingCells,
            int rejectedCount,
            int outOfBoundsCount,
            int invalidFitnessCount,
            int occupiedCells,
            int cellCount,
            float coverage,
            float qdScore,
            float bestEvaluatedFitness,
            float bestEliteFitness,
            TimeSpan totalElapsed,
            TimeSpan askElapsed,
            TimeSpan evaluateElapsed,
            TimeSpan tellElapsed)
        {
            Iteration = iteration;
            InsertedNewCells = insertedNewCells;
            ReplacedExistingCells = replacedExistingCells;
            RejectedCount = rejectedCount;
            OutOfBoundsCount = outOfBoundsCount;
            InvalidFitnessCount = invalidFitnessCount;
            OccupiedCells = occupiedCells;
            CellCount = cellCount;
            Coverage = coverage;
            QdScore = qdScore;
            BestEvaluatedFitness = bestEvaluatedFitness;
            BestEliteFitness = bestEliteFitness;
            TotalElapsed = totalElapsed;
            AskElapsed = askElapsed;
            EvaluateElapsed = evaluateElapsed;
            TellElapsed = tellElapsed;
        }

        public int Iteration { get; }
        public int InsertedNewCells { get; }
        public int ReplacedExistingCells { get; }
        public int RejectedCount { get; }
        public int OutOfBoundsCount { get; }
        public int InvalidFitnessCount { get; }
        public int OccupiedCells { get; }
        public int CellCount { get; }
        public float Coverage { get; }
        public float QdScore { get; }

        /// <summary>
        ///     Strongest fitness ever produced by the evaluator across all iterations,
        ///     regardless of archive admission. Reflects the raw search progress of the
        ///     emitter and the evaluator working together.
        /// </summary>
        public float BestEvaluatedFitness { get; }

        /// <summary>
        ///     Strongest fitness currently held in the archive. Monotone non-decreasing
        ///     in the current implementation because <c>Insert</c> only replaces when the
        ///     new fitness is strictly higher than the existing elite.
        /// </summary>
        public float BestEliteFitness { get; }

        /// <summary>
        ///     Wall-clock duration of one full iteration (Ask + Evaluate + Tell).
        ///     Populated by <c>MapElites.RunIteration</c>; for callers that drive
        ///     Ask/Evaluate/Tell manually via <see cref="MapElites{TContext}.Tell"/>,
        ///     this is <see cref="TimeSpan.Zero"/>.
        /// </summary>
        public TimeSpan TotalElapsed { get; }

        public TimeSpan AskElapsed { get; }
        public TimeSpan EvaluateElapsed { get; }
        public TimeSpan TellElapsed { get; }
    }
}