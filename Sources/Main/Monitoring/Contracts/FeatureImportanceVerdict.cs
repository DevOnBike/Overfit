namespace DevOnBike.Overfit.Monitoring.Contracts
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