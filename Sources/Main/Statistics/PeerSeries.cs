// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// One member of a peer group — a pod of a Deployment, a node of a pool, a replica of a StatefulSet —
    /// together with its observation window.
    /// </summary>
    /// <param name="Name">Identifier carried into the finding so a report can name the deviating member.</param>
    /// <param name="Values">The signal's observations, ascending in time. Non-finite samples are skipped.</param>
    /// <param name="Work">
    /// Optional units of work over the same timestamps (requests served, bytes processed), used to normalise
    /// <paramref name="Values"/> into cost-per-unit. Required for load-sensitive signals — see
    /// <see cref="PeerSignalKind"/> — because otherwise a peer that legitimately handles three times the
    /// traffic looks identical to a broken one.
    /// </param>
    public readonly record struct PeerSeries(
        string Name,
        ReadOnlyMemory<double> Values,
        ReadOnlyMemory<double> Work = default);
}
