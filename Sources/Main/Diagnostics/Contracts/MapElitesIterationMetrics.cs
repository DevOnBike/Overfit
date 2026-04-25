// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;

namespace DevOnBike.Overfit.Evolutionary.Runtime
{
    public readonly struct MapElitesIterationMetrics
    {
        public MapElitesIterationMetrics(
            int iteration,
            int insertedNewCells,
            int replacedExistingCells,
            int rejectedCount,
            int outOfBoundsCount,
            int occupiedCells,
            int cellCount,
            float coverage,
            float qdScore,
            float bestFitness)
        {
            Iteration = iteration;
            InsertedNewCells = insertedNewCells;
            ReplacedExistingCells = replacedExistingCells;
            RejectedCount = rejectedCount;
            OutOfBoundsCount = outOfBoundsCount;
            OccupiedCells = occupiedCells;
            CellCount = cellCount;
            Coverage = coverage;
            QdScore = qdScore;
            BestFitness = bestFitness;
        }

        public int Iteration { get; }
        public int InsertedNewCells { get; }
        public int ReplacedExistingCells { get; }
        public int RejectedCount { get; }
        public int OutOfBoundsCount { get; }
        public int OccupiedCells { get; }
        public int CellCount { get; }
        public float Coverage { get; }
        public float QdScore { get; }
        public float BestFitness { get; }
        public TimeSpan TotalElapsed { get; }
        public TimeSpan AskElapsed { get; }
        public TimeSpan EvaluateElapsed { get; }
        public TimeSpan TellElapsed { get; }
    }
}