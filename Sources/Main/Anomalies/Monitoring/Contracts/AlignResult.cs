// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>
    ///     Result of a single alignment pass.
    ///     Windows[i] and PodIndex[i] always refer to the same pod.
    /// </summary>
    public sealed class AlignResult
    {
        /// <summary>Aligned windows — one per pod, same order as PodIndex.</summary>
        public List<RawPodWindow> Windows { get; private set; } = [];

        /// <summary>
        ///     Pod identity for each window. From this point forward, pod name lives
        ///     only here — not inside the data arrays.
        /// </summary>
        public List<PodKey> PodIndex { get; private set; } = [];
    }

}