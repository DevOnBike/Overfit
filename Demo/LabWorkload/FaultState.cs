// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LabWorkload
{
    /// <summary>
    /// The faults this replica is currently pretending to have, changeable at runtime.
    ///
    /// <para><b>Why a control plane rather than environment variables alone.</b> Injecting a fault by editing
    /// a manifest costs a pod restart, and a restart is itself a fault: the working set ramps from cold, the
    /// restart counter moves, and the new pod has no history inside the evaluation window. Every one of those
    /// is a signal the guard reacts to, so redeploying to test a leak means testing a leak <i>and</i> a
    /// restart and being unable to say which produced what. This project has already lost measurements to
    /// exactly that confound.</para>
    ///
    /// <para>Flipping a fault on a running pod changes one thing. The window keeps its history, no counter
    /// moves, and the detector's reaction is attributable — which is the difference between watching an
    /// experiment and watching a rollout.</para>
    ///
    /// <para>Fields are volatile rather than locked: every one is a single word written by one request thread
    /// and read by another, torn reads are not possible on a <c>double</c> at this width on the platforms
    /// this runs on, and a lock around a knob that a test flips once per minute would be ceremony.</para>
    /// </summary>
    internal sealed class FaultState
    {
        private double _latencyMs;
        private double _latencyJitter;
        private double _stallProbability;
        private double _stallSeconds;
        private double _errorRate;
        private double _leakBytesPerSecond;
        private double _cpuBurnMs;

        public FaultState(FaultProfile initial)
        {
            ArgumentNullException.ThrowIfNull(initial);

            _latencyMs = initial.LatencyMs;
            _latencyJitter = initial.LatencyJitter;
            _stallProbability = initial.StallProbability;
            _stallSeconds = initial.StallSeconds;
            _errorRate = initial.ErrorRate;
            _leakBytesPerSecond = initial.MemoryLeakBytesPerSecond;
            _cpuBurnMs = initial.CpuBurnMs;

            Role = initial.Role;
        }

        public string Role { get; }

        public double LatencyMs
        {
            get => Volatile.Read(ref _latencyMs);
            set => Volatile.Write(ref _latencyMs, Math.Max(0.0, value));
        }

        public double LatencyJitter
        {
            get => Volatile.Read(ref _latencyJitter);
            set => Volatile.Write(ref _latencyJitter, Math.Clamp(value, 0.0, 2.0));
        }

        public double StallProbability
        {
            get => Volatile.Read(ref _stallProbability);
            set => Volatile.Write(ref _stallProbability, Math.Clamp(value, 0.0, 1.0));
        }

        public double StallSeconds
        {
            get => Volatile.Read(ref _stallSeconds);
            set => Volatile.Write(ref _stallSeconds, Math.Max(0.0, value));
        }

        public double ErrorRate
        {
            get => Volatile.Read(ref _errorRate);
            set => Volatile.Write(ref _errorRate, Math.Clamp(value, 0.0, 1.0));
        }

        public double LeakBytesPerSecond
        {
            get => Volatile.Read(ref _leakBytesPerSecond);
            set => Volatile.Write(ref _leakBytesPerSecond, Math.Max(0.0, value));
        }

        public double CpuBurnMs
        {
            get => Volatile.Read(ref _cpuBurnMs);
            set => Volatile.Write(ref _cpuBurnMs, Math.Max(0.0, value));
        }

        /// <summary>Back to a good replica, without a restart.</summary>
        public void Clear()
        {
            StallProbability = 0.0;
            ErrorRate = 0.0;
            LeakBytesPerSecond = 0.0;
            CpuBurnMs = 0.0;
        }

        public string Describe()
        {
            var faults = new List<string>(4);

            if (StallProbability > 0.0)
            {
                faults.Add($"stall {StallProbability:P0}@{StallSeconds:F1}s");
            }

            if (ErrorRate > 0.0)
            {
                faults.Add($"errors {ErrorRate:P0}");
            }

            if (LeakBytesPerSecond > 0.0)
            {
                faults.Add($"leak {LeakBytesPerSecond / 1_000_000.0:F1}MB/s");
            }

            if (CpuBurnMs > 0.0)
            {
                faults.Add($"cpu {CpuBurnMs:F0}ms/req");
            }

            return faults.Count == 0
                ? $"healthy (latency {LatencyMs:F0}ms +-{LatencyJitter:P0})"
                : $"{string.Join(", ", faults)} (latency {LatencyMs:F0}ms +-{LatencyJitter:P0})";
        }
    }
}
