// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    /// <summary>Importance verdict for a single feature dimension.</summary>
    public enum FeatureImportanceVerdict
    {
        /// <summary>
        /// Feature importance is statistically greater than the best shadow feature.
        /// The autoencoder relies on this feature for reconstruction.
        /// </summary>
        Confirmed,

        /// <summary>
        /// Inconclusive after all iterations. The feature may or may not be relevant.
        /// Consider keeping it or running more iterations.
        /// </summary>
        Tentative,

        /// <summary>
        /// Feature importance is not statistically different from random shadow features.
        /// Safe to remove from the model without meaningful loss of detection quality.
        /// </summary>
        Rejected
    }
}