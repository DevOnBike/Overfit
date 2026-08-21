// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using ILGPU.Runtime;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// One timed arm. A GPU arm can ONLY be built through <see cref="Gpu"/>, which takes the accelerator
    /// and captures its <c>Synchronize</c> as the barrier; <see cref="TimeOnce"/> then invokes that
    /// barrier before it stops the clock.
    /// <para>
    /// The constructor is private on purpose. An ILGPU kernel launch is asynchronous, so a timer stopped
    /// straight after the launch measures the enqueue and reports a spectacular, meaningless speedup.
    /// Section 3.5 rule 4 of the plan names that as the single most likely defect in the whole probe.
    /// Putting the synchronise inside the timing helper — rather than at each call site — makes it
    /// impossible to write a GPU arm that forgets it.
    /// </para>
    /// </summary>
    internal sealed class Arm
    {
        private readonly Action _body;
        private readonly Action? _barrier;
        private readonly Action? _before;
        private readonly Action? _after;

        private Arm(string name, Action body, Action? barrier, Action? before, Action? after)
        {
            Name = name;
            _body = body;
            _barrier = barrier;
            _before = before;
            _after = after;
        }

        public string Name { get; }

        /// <summary>True when this arm ends at a device barrier.</summary>
        public bool IsDevice => _barrier is not null;

        /// <summary>
        /// A host arm. Nothing is enqueued, so there is nothing to wait for.
        /// <paramref name="before"/> and <paramref name="after"/> run OUTSIDE the clock and exist for the
        /// setup an arm needs per repetition but which is not the thing under measurement — seeding an
        /// output gradient, or returning a graph arena.
        /// </summary>
        public static Arm Cpu(string name, Action body, Action? before = null, Action? after = null)
            => new(name, body, null, before, after);

        /// <summary>
        /// A device arm. The accelerator is required rather than optional precisely so that the barrier
        /// cannot be omitted.
        /// </summary>
        public static Arm Gpu(string name, Accelerator accelerator, Action body)
        {
            ArgumentNullException.ThrowIfNull(accelerator);
            return new Arm(name, body, accelerator.Synchronize, before: null, after: null);
        }

        /// <summary>Runs the body once and returns the wall time in milliseconds, barrier included.</summary>
        public double TimeOnce()
        {
            _before?.Invoke();

            var start = Stopwatch.GetTimestamp();
            _body();
            _barrier?.Invoke();
            var elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;

            _after?.Invoke();
            return elapsed;
        }
    }
}
