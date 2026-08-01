// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;

namespace DevOnBike.Overfit.LabWorkload
{
    /// <summary>
    /// What this replica pretends to be wrong with, read from the environment so one image serves every pod
    /// in the lab and a manifest decides which of them is broken.
    ///
    /// <para><b>Why a synthetic workload replaced the real inference server in the lab.</b> The server was a
    /// poor instrument for three measured reasons. It loads a 0.5B model, so a 32-core node fits four
    /// replicas at faithful load and the between-pod statistic stays a range over three draws. It offers
    /// exactly one injectable fault — a CPU limit — and that fault turned out not to be what anyone assumed:
    /// the throttled replica does not run uniformly slower, it <b>stalls for minutes and then catches up</b>,
    /// which was discovered by accident when a pre-flight check read its request rate as zero. And its
    /// latency depends on how busy the machine is, so the operating point drifted between runs and cost
    /// three of them.</para>
    ///
    /// <para><b>The point of doing it this way is ground truth.</b> An afternoon went into inferring, from
    /// ranges and interquartile spreads and drawdowns, whether the lab's memory was a sawtooth or a monotone
    /// climb — a question this file answers by construction. A generator calibrated against a reference whose
    /// shape is known is a different kind of exercise from one calibrated against a recording that has to be
    /// decoded first.</para>
    ///
    /// <para>Every knob defaults to healthy. A pod with no environment set is a good replica, which is what
    /// most of them should be — a peer group where everything is broken has no norm to deviate from.</para>
    /// </summary>
    internal sealed record FaultProfile
    {
        /// <summary>Median service time before any fault is applied.</summary>
        public double LatencyMs { get; init; } = 40.0;

        /// <summary>
        /// Full width of the multiplicative scatter around <see cref="LatencyMs"/>, as a fraction.
        ///
        /// <para>Uniform, so the range is almost exactly twice the interquartile spread. That relationship is
        /// worth having under control: it is the diagnostic that separates a missing mechanism from a wrong
        /// number, and here it is a setting rather than something to be recovered from data.</para>
        /// </summary>
        public double LatencyJitter { get; init; } = 0.30;

        /// <summary>
        /// Chance a request stalls, and for how long — a fault the real lab produced and nobody designed.
        ///
        /// <para>Modelled separately from latency because it is a different shape: a stall does not scale the
        /// service time, it adds a large fixed wait to whatever requests are in flight. That distinction
        /// matters downstream — an additive delay moves a p50 by far more, in relative terms, than it moves a
        /// p99, which is the asymmetry a multiplicative fault cannot produce.</para>
        /// </summary>
        public double StallProbability { get; init; }

        /// <inheritdoc cref="StallProbability"/>
        public double StallSeconds { get; init; } = 5.0;

        /// <summary>Share of requests answered 500.</summary>
        public double ErrorRate { get; init; }

        /// <summary>
        /// Bytes retained per second, never released — a leak with a rate somebody chose.
        ///
        /// <para>The one fault the trend family exists for, and the one the real lab could not produce at
        /// all. Its absolute rate is what makes a measured floor meaningful: "256 MiB over a window" can be
        /// checked against a leak of a known size rather than against a guess.</para>
        /// </summary>
        public double MemoryLeakBytesPerSecond { get; init; }

        /// <summary>Milliseconds of actual CPU burned per request, on top of the sleep.</summary>
        public double CpuBurnMs { get; init; }

        /// <summary>
        /// Free-form role label, exported as a metric label.
        ///
        /// <para>For the leader/follower case: three replicas of one image where one holds a lease and
        /// legitimately behaves differently. The guard cannot infer that a difference is by design, so the
        /// intended answer is that the cluster already carries the fact and the guard is told which label to
        /// read.</para>
        /// </summary>
        public string Role { get; init; } = "member";

        public static FaultProfile FromEnvironment()
        {
            return new FaultProfile
            {
                LatencyMs = Number("LAB_LATENCY_MS", 40.0),
                LatencyJitter = Number("LAB_LATENCY_JITTER", 0.30),
                StallProbability = Number("LAB_STALL_PROBABILITY", 0.0),
                StallSeconds = Number("LAB_STALL_SECONDS", 5.0),
                ErrorRate = Number("LAB_ERROR_RATE", 0.0),
                MemoryLeakBytesPerSecond = Number("LAB_MEMORY_LEAK_BYTES_PER_SECOND", 0.0),
                CpuBurnMs = Number("LAB_CPU_BURN_MS", 0.0),
                Role = Environment.GetEnvironmentVariable("LAB_ROLE") is { Length: > 0 } role
                    ? role
                    : "member",
            };
        }

        /// <summary>One line naming every fault that is on, so a pod's logs say what it is pretending to be.</summary>
        public string Describe()
        {
            var faults = new List<string>(5);

            if (StallProbability > 0.0)
            {
                faults.Add($"stalls {StallProbability:P1} of requests for {StallSeconds:F1}s");
            }

            if (ErrorRate > 0.0)
            {
                faults.Add($"fails {ErrorRate:P1} of requests");
            }

            if (MemoryLeakBytesPerSecond > 0.0)
            {
                faults.Add($"leaks {MemoryLeakBytesPerSecond / 1_000_000.0:F1} MB/s");
            }

            if (CpuBurnMs > 0.0)
            {
                faults.Add($"burns {CpuBurnMs:F0}ms CPU per request");
            }

            return faults.Count == 0
                ? "healthy"
                : string.Join(", ", faults);
        }

        private static double Number(string name, double fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
                   && value >= 0.0
                ? value
                : fallback;
        }
    }
}
