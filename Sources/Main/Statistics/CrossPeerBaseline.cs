// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// What a peer group was doing <b>together</b> at each instant — the median across sibling replicas,
    /// sample by sample. Feed it to <see cref="TrendDetector.Detect"/> as the expectation and the per-pod test
    /// runs on what that pod did <i>differently</i>.
    ///
    /// <para><b>The problem it solves, measured on the cluster lab.</b> Over a loaded twelve-minute window,
    /// ten of the eleven findings on healthy replicas were the same falling latency trend on all three pods at
    /// once — p50, p95 and p99, tau between −0.46 and −0.72. The server was warming up: JIT, caches, pools.
    /// Latency really did fall, on every replica, together. Each finding was arithmetically correct and named
    /// a pod, and not one of them was about that pod.</para>
    ///
    /// <para><b>A signal moving the same way on every replica is the workload or the environment, not a
    /// fault.</b> <see cref="PeerGroupOutlierDetector"/> rejects this by construction — if everyone moves
    /// together nobody is an outlier — but <see cref="TrendDetector"/> sees one series at a time and has no
    /// way to know the others did the same thing.</para>
    ///
    /// <para><b>This does not suppress fleet-wide drift, and suppressing it would be the wrong fix.</b>
    /// Catching what affects every replica at once is the one thing the trend family can do and the peer
    /// comparison structurally cannot: three replicas all leaking memory is a real fault and a peer test is
    /// blind to it. So the group is <i>decomposed</i> rather than filtered — run the trend on this common
    /// component to ask "is the deployment drifting?", and on each pod's residual to ask "is this pod drifting
    /// differently?". Both questions keep their answer; what changes is that the first one is reported once,
    /// about the workload, instead of once per pod.</para>
    ///
    /// <para><b>The median needs no leave-one-out.</b> The peer detector excludes the candidate from its own
    /// baseline because a mean would be dragged by it. A median with three or more members cannot be moved
    /// past the middle value by a single deviating one, so the common component is already the healthy
    /// majority's behaviour. That protection degrades in the same way and at the same bound as everywhere else
    /// here: once more than a third of the group deviates together, they <i>are</i> the majority, and the
    /// decomposition will call their movement common. That is not a defect to fix at this layer — it is the
    /// definition of common mode.</para>
    /// </summary>
    public static class CrossPeerBaseline
    {
        /// <summary>
        /// Members required before a common component means anything. Below three there is no majority to
        /// take a median of: with two, the median is their average and each pod's residual is just half the
        /// difference to the other, with the sign flipped — every deviation would appear twice, once in each
        /// direction, and neither would name a culprit.
        /// </summary>
        public const int MinimumPeers = 3;

        /// <summary>Scratch the caller must supply: one slot per peer.</summary>
        public static int RequiredScratchLength(int peerCount)
        {
            return peerCount < 0 ? 0 : peerCount;
        }

        /// <summary>
        /// Writes the across-peer median at each sample index into <paramref name="expectation"/>.
        ///
        /// <para>An index where fewer than <see cref="MinimumPeers"/> members reported a finite value yields
        /// <see cref="double.NaN"/>. That is deliberate and it is not the same as zero: with one or two
        /// reporters the "median" is dominated by whoever happened to be scraped, and subtracting it would
        /// flatten that pod's own residual to nothing — silencing exactly the pod whose data survived. A NaN
        /// expectation makes the sample NaN downstream, and every detector here drops those.</para>
        /// </summary>
        /// <param name="peers">Group members. Series are read up to <paramref name="expectation"/>'s length;
        /// a member shorter than that contributes nothing beyond its end rather than being an error, because a
        /// pod younger than the window is a normal condition.</param>
        /// <param name="expectation">Receives one value per sample index.</param>
        /// <param name="scratch">At least <see cref="RequiredScratchLength"/> entries.</param>
        /// <returns><c>false</c> when the group is too small to have a common component at all.</returns>
        public static bool TryBuild(
            IReadOnlyList<PeerSeries> peers,
            Span<double> expectation,
            Span<double> scratch)
        {
            ArgumentNullException.ThrowIfNull(peers);

            if (peers.Count < MinimumPeers)
            {
                return false;
            }

            if (scratch.Length < RequiredScratchLength(peers.Count))
            {
                throw new ArgumentException(
                    $"Scratch holds {scratch.Length} entries; {RequiredScratchLength(peers.Count)} are needed, "
                    + "one per peer.",
                    nameof(scratch));
            }

            for (var i = 0; i < expectation.Length; i++)
            {
                var reporting = 0;

                for (var p = 0; p < peers.Count; p++)
                {
                    var values = peers[p].Values.Span;

                    if (i >= values.Length)
                    {
                        continue;
                    }

                    var value = values[i];

                    if (!double.IsFinite(value))
                    {
                        continue;
                    }

                    scratch[reporting] = value;
                    reporting++;
                }

                expectation[i] = reporting < MinimumPeers
                    ? double.NaN
                    : MedianSelector.MedianInPlace(scratch.Slice(0, reporting));
            }

            return true;
        }
    }
}
