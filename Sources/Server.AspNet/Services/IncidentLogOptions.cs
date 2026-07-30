// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Server.AspNet.Services
{
    /// <summary>
    /// How <see cref="LoggerIncidentSink"/> routes what it emits.
    /// </summary>
    /// <param name="Level">
    /// Level for incidents — the rows an operator might act on.
    ///
    /// <para><b>A log level is a routing decision in most deployments</b>, and Warning or above reaches
    /// somebody. Until the false-positive rate has been measured on the cluster this will actually run
    /// against, that somebody will be reached for nothing, and the first operator who filters this channel
    /// out will not put it back.</para>
    /// </param>
    /// <param name="FindingLevel">
    /// Level for the evidence inside an incident. Separate from <paramref name="Level"/> because they answer
    /// different questions and are counted differently: incidents measure how many things went wrong,
    /// findings measure how noisy the detector is. Conflating the two has already produced one wrong
    /// conclusion here — 1193 findings were 254 incidents, and a plan built on the finding count was refuted
    /// by an ablation.
    /// </param>
    /// <param name="IncludeFindings">
    /// Whether to emit the evidence at all. Off makes the channel one line per incident, which is what a
    /// paging path wants; on is what an investigation wants.
    /// </param>
    public readonly record struct IncidentLogOptions(
        LogLevel Level,
        LogLevel FindingLevel,
        bool IncludeFindings)
    {
        /// <summary>
        /// The default, and the one to start with: everything at <see cref="LogLevel.Information"/>, evidence
        /// included. Counts and records; wakes nobody.
        ///
        /// <para>This is the shape of the rollout this guard needs — run it against a real cluster, measure
        /// what it says, and let the operator label which of it was real. That produces the false-positive
        /// rate on somebody else's data <i>and</i> the labels the learned stack has never had, as a byproduct
        /// of the first step rather than as a prerequisite for it.</para>
        /// </summary>
        public static IncidentLogOptions Shadow =>
            new(LogLevel.Information, LogLevel.Information, IncludeFindings: true);

        /// <summary>
        /// Once the rate has been measured and accepted: incidents at <see cref="LogLevel.Warning"/> so they
        /// route, evidence at <see cref="LogLevel.Debug"/> so it is there when someone goes looking.
        /// </summary>
        public static IncidentLogOptions Routed =>
            new(LogLevel.Warning, LogLevel.Debug, IncludeFindings: true);

        /// <summary>Incidents only — one line each, no evidence.</summary>
        public static IncidentLogOptions Quiet =>
            new(LogLevel.Warning, LogLevel.Debug, IncludeFindings: false);
    }
}
