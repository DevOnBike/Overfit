// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Xunit.Abstractions;
using Xunit.Sdk;

[assembly: Xunit.TestFramework(
    "DevOnBike.Overfit.Tests.MeasurementExclusion", "DevOnBike.Overfit.Tests")]

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// A test run and a benchmark run may not share this machine.
    ///
    /// <para><b>Symmetric, and both directions protect the same thing.</b> A suite that starts during a
    /// benchmark saturates thirty-two cores for half a minute and lands inside the benchmark's samples; a
    /// benchmark that starts during a suite does the same in reverse. Either way the number that comes out is
    /// wrong and says nothing about being wrong, which is the failure this repository has paid for more than
    /// once — the A/B discipline in <c>CLAUDE.md</c> exists because cross-process drift on this box has
    /// already reached about 30%.</para>
    ///
    /// <para><b>Why refusing beats detecting.</b> The obvious alternative is to let both run and mark the
    /// benchmark's output as suspect. That is strictly worse: it spends the forty minutes first and tells you
    /// afterwards. A refusal at second zero costs nothing and is unambiguous.</para>
    ///
    /// <para><b>Why blocking a test run is acceptable here, having argued that it is not.</b> The objection
    /// is that correctness work should never wait on a performance claim. It holds in general and does not
    /// hold on one machine that is also the measurement instrument: a benchmark is started deliberately, runs
    /// for a bounded time, and is the expensive thing to redo. Waiting for it costs minutes; corrupting it
    /// costs the run and, worse, may not be noticed.</para>
    ///
    /// <para>Registered through <c>[assembly: TestFramework]</c> rather than a fixture, because this has to
    /// hold for the whole assembly however the run was launched — <c>dotnet test</c>, an IDE runner, or a
    /// debugger attached to one test.</para>
    /// </summary>
    public sealed class MeasurementExclusion : XunitTestFramework
    {
        /// <summary>
        /// The lock both sides take. One name, so either side blocks the other; <c>Global\</c> so a run
        /// started from another terminal or another user session is caught, which a session-scoped mutex
        /// would miss.
        /// </summary>
        internal const string MutexName = @"Global\DevOnBike.Overfit.MachineMeasurement";

        private readonly Mutex? _held;

        public MeasurementExclusion(IMessageSink messageSink)
            : base(messageSink)
        {
            // Opt-out for the case this cannot anticipate: a CI agent that runs suites concurrently by
            // design, where the mutex would serialise unrelated jobs on one host. Named rather than
            // silent, so switching it off is a decision somebody wrote down.
            if (Environment.GetEnvironmentVariable("OVERFIT_ALLOW_CONCURRENT_MEASUREMENT") == "1")
            {
                return;
            }

            var mutex = new Mutex(initiallyOwned: false, MutexName, out _);
            var acquired = false;

            try
            {
                acquired = mutex.WaitOne(TimeSpan.Zero);
            }
            catch (AbandonedMutexException)
            {
                // A previous run died without releasing it. Nobody is measuring; the lock is ours.
                acquired = true;
            }

            if (!acquired)
            {
                mutex.Dispose();

                throw new InvalidOperationException(
                    "An Overfit benchmark is running on this machine, so the test suite is refusing to "
                    + "start: thirty-two cores of test load inside a benchmark's sampling window produces a "
                    + "wrong number that looks like a measurement. Wait for it to finish, or set "
                    + "OVERFIT_ALLOW_CONCURRENT_MEASUREMENT=1 if you know the two are not sharing a box.");
            }

            _held = mutex;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing && _held is not null)
            {
                _held.ReleaseMutex();
                _held.Dispose();
            }

            base.Dispose(disposing);
        }
    }
}
