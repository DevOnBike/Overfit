// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    public enum DropStrategy
    {
        /// <summary>Retains the column with the lower index — fast and deterministic.</summary>
        KeepFirst,

        /// <summary>
        ///     Retains the column more strongly correlated with the Target — slower,
        ///     but provides superior predictive feature selection.
        /// </summary>
        KeepHigherTargetCorrelation
    }
}