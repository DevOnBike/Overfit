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
    /// this runs on, and a lock around a knob that a test flips once per minute would be ceremony. The OOM
    /// cancellation source is the exception — it is swapped with <see cref="Interlocked"/> because two
    /// concurrent injections must not each believe they own the running allocation.</para>
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
        private double _throwRate;

        /// <summary>
        /// Non-null while an OOM allocation is running. It exists because the first version had no way to
        /// stop one: the loop was <c>while (true)</c> in a detached task, so <see cref="Clear"/> returned
        /// "healthy" while the thread kept allocating, and only killing the pod could end it — which is the
        /// one thing the control plane exists to avoid.
        /// </summary>
        private CancellationTokenSource? _oomAllocation;

        /// <summary>
        /// Non-null while threads are deliberately fighting over a lock. Its own source rather than sharing
        /// the OOM one: an OOM allocation ends the process, contention does not, and a single token would
        /// make <see cref="Clear"/> unable to stop one without the other having been started.
        /// </summary>
        private CancellationTokenSource? _contention;

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

        public string Role
        {
            get;
        }

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

        /// <summary>
        /// Share of requests that throw an exception <b>which is then caught</b>, so the request still
        /// succeeds. Deliberately separate from <see cref="ErrorRate"/>: that one returns a 500 without
        /// throwing, and the difference between the two is exactly what the exceptions channel exists to
        /// see.
        /// </summary>
        public double ThrowRate
        {
            get => Volatile.Read(ref _throwRate);
            set => Volatile.Write(ref _throwRate, Math.Clamp(value, 0.0, 1.0));
        }

        /// <summary>True while an OOM allocation is in flight, so <see cref="Describe"/> cannot claim health
        /// during the seconds before the kernel kills this process.</summary>
        public bool IsAllocatingToOom
            => Volatile.Read(ref _oomAllocation) is { IsCancellationRequested: false };

        /// <summary>
        /// Starts an OOM allocation and returns the token that stops it. Any allocation already running is
        /// cancelled first: two overlapping injections would otherwise race to the limit and leave one loop
        /// unreachable.
        /// </summary>
        // OVERFIT040 — synchronous by design. The call the rule points at is
        // CancellationTokenSource.Cancel(), whose async sibling CancelAsync exists for ONE reason: Cancel
        // runs registered callbacks on the calling thread, so a registration that blocks blocks the caller.
        // There are no registrations here to run — grep the project: nothing calls Token.Register, and the
        // fault loops poll IsCancellationRequested. The only registrations on these sources are the ones
        // Task.Run / Task.Factory.StartNew make internally, which flip task state and return. So Cancel()
        // has no user code to execute and holds no thread; this is not the "synchronous island" the rule
        // describes, and CancelAsync would only push a flag flip onto the pool.
        //
        // The shape also constrains it: these methods return a CancellationToken (not a task) to synchronous
        // minimal-API handlers, so an async signature would make the whole fault-injection surface
        // asynchronous in order to set a flag.
#pragma warning disable OVERFIT040
        public CancellationToken BeginOomAllocation()
#pragma warning restore OVERFIT040
        {
            var fresh = new CancellationTokenSource();
            var previous = Interlocked.Exchange(ref _oomAllocation, fresh);

            previous?.Cancel();
            previous?.Dispose();

            return fresh.Token;
        }

        /// <summary>True while threads are contending, so <see cref="Describe"/> cannot claim health.</summary>
        public bool IsContending
            => Volatile.Read(ref _contention) is { IsCancellationRequested: false };

        /// <summary>
        /// Starts lock contention and returns the token that stops it, cancelling any run already in
        /// progress so two injections cannot leave one set of threads unreachable.
        /// </summary>
        // OVERFIT040 — same constraint as BeginOomAllocation above: Cancel() has no registered callbacks to
        // run here, so it holds no thread.
#pragma warning disable OVERFIT040
        public CancellationToken BeginContention()
#pragma warning restore OVERFIT040
        {
            var fresh = new CancellationTokenSource();
            var previous = Interlocked.Exchange(ref _contention, fresh);

            previous?.Cancel();
            previous?.Dispose();

            return fresh.Token;
        }

        /// <summary>Back to a good replica, without a restart.</summary>
        // OVERFIT040 — same constraint as BeginOomAllocation above: Cancel() has no registered callbacks to
        // run here, so it holds no thread. Clear() is also the undo path POST /fault/clear calls
        // synchronously, and it must leave the fault state consistent before that handler returns.
#pragma warning disable OVERFIT040
        public void Clear()
#pragma warning restore OVERFIT040
        {
            StallProbability = 0.0;
            ErrorRate = 0.0;
            LeakBytesPerSecond = 0.0;
            CpuBurnMs = 0.0;
            ThrowRate = 0.0;

            var allocation = Interlocked.Exchange(ref _oomAllocation, null);

            allocation?.Cancel();
            allocation?.Dispose();

            var contention = Interlocked.Exchange(ref _contention, null);

            contention?.Cancel();
            contention?.Dispose();
        }

        public string Describe()
        {
            var faults = new List<string>(5);

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

            if (IsAllocatingToOom)
            {
                faults.Add("oom (allocating)");
            }

            if (IsContending)
            {
                faults.Add("lock contention");
            }

            if (ThrowRate > 0.0)
            {
                faults.Add($"throwing {ThrowRate:P0}");
            }

            return faults.Count == 0
                ? $"healthy (latency {LatencyMs:F0}ms +-{LatencyJitter:P0})"
                : $"{string.Join(", ", faults)} (latency {LatencyMs:F0}ms +-{LatencyJitter:P0})";
        }
    }
}
