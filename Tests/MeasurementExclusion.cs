// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Xunit.v3;

// xunit v3 takes the framework as a Type rather than as two strings. That is a real improvement here and
// not just a signature change: the old form named the class and its assembly in text, so a rename or a
// move compiled cleanly and silently fell back to the default framework — this guard's exact failure
// mode, and one already paid for once in the note on Environment.Exit below.
[assembly: Xunit.TestFramework(typeof(DevOnBike.Overfit.Tests.MeasurementExclusion))]

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
    ///
    /// <para><b>What the xunit v3 migration changed, measured on 2026-08-11 by holding the mutex externally
    /// and running both ways.</b> The protection is intact: the suite does not run. The DIAGNOSIS is not,
    /// and only through <c>dotnet test</c>.</para>
    ///
    /// <list type="bullet">
    /// <item><description>Running the built executable directly — v3 makes the test project an exe — is
    /// exactly right: exit code <b>2</b> and the full message below.</description></item>
    /// <item><description>Through <c>dotnet test</c>, the VSTest bridge launches that exe as a CHILD and
    /// speaks JSON to it. <c>Environment.Exit(2)</c> lands in the middle of the bridge's very first probe,
    /// so the operator sees <c>Catastrophic failure: Test process did not return valid JSON (non-object)</c>
    /// plus "no tests available", and the process reports <b>1</b>, not 2. The message never appears.
    /// </description></item>
    /// </list>
    ///
    /// <para>So a script keying on exit code 2 no longer works through <c>dotnet test</c>, and a human who
    /// hits this reads something unintelligible instead of the sentence written for them. Recorded rather
    /// than papered over: a refusal nobody can interpret is only half a guard. Tracked as <c>XC-17</c>.</para>
    ///
    /// <para><b>So the refusal writes itself down instead.</b> Since 2026-08-14 it leaves
    /// <c>Tests/bin/measurement-refusal.txt</c> — see <see cref="MeasurementRefusalMarker"/> — naming the
    /// lock, the timestamp, the run that was refused and every live process that could be holding it. The
    /// console cannot carry the message; a file can, and a script can read it. This does not fix
    /// <c>XC-17</c>'s exit code, and is not meant to: it makes the reason legible, which is the half that
    /// cost an hour on 2026-08-13 (<c>XC-42</c>).</para>
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

        // v3 constructs the framework with no arguments; the IMessageSink the v2 base class took is gone.
        // Nothing in this guard used it.
        public MeasurementExclusion()
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

                // Ending the process rather than throwing, and the difference was measured rather than
                // assumed: an exception from this constructor is CAUGHT BY THE RUNNER, which then falls
                // back to the default framework and runs the whole suite. Verified on 2026-08-03 by
                // holding the mutex externally — the suite passed, and a marker file proved the
                // constructor had run and its exception had been discarded. A guard whose refusal is
                // swallowed is worse than no guard, because the green result looks like proof.
                //
                // Exit code 2 is the same "someone else is measuring" code the benchmark host uses, so a
                // script can tell it from a test failure.
                Console.Error.WriteLine(
                    "An Overfit benchmark is running on this machine, so the test suite is refusing to "
                    + "start: thirty-two cores of test load inside a benchmark's sampling window produces a "
                    + "wrong number that looks like a measurement. Wait for it to finish, or set "
                    + "OVERFIT_ALLOW_CONCURRENT_MEASUREMENT=1 if you know the two are not sharing a box.");

                // Written down because the sentence above does not survive the VSTest bridge — see XC-17,
                // and see XC-42 for the hour it cost when it did not. The marker names the live holder,
                // which is the fact the console cannot deliver.
                var marker = MeasurementRefusalMarker.Write(MutexName);

                if (marker is not null)
                {
                    Console.Error.WriteLine($"Details written to {marker}");
                }

                Console.Error.Flush();

                Environment.Exit(2);
            }

            _held = mutex;

            ReleaseOnExit();
        }

        /// <summary>
        /// Releases the lock when the process ends, rather than from a disposal override.
        ///
        /// <para><b>Deliberately not tied to the base class's lifetime.</b> `XunitTestFramework` exposes no
        /// `Dispose(bool)` to override in xUnit 2.9, and guessing at a base class's disposal shape is how a
        /// guard ends up released at a moment nobody intended. A test run's lifetime <i>is</i> the process
        /// lifetime, so the process exiting is the correct and simplest signal.</para>
        ///
        /// <para>If the process dies without running this — killed, crashed — the operating system releases
        /// the mutex anyway and the next acquirer sees `AbandonedMutexException`, which both sides already
        /// treat as "nobody is measuring, the lock is ours". There is no state to leave stale.</para>
        /// </summary>
        private void ReleaseOnExit()
        {
            AppDomain.CurrentDomain.ProcessExit += (_, _) =>
            {
                try
                {
                    _held?.ReleaseMutex();
                }
                catch (ApplicationException)
                {
                    // Released already, or owned by another thread. Nothing to do and nothing to report:
                    // the process is ending and the OS is about to release it regardless.
                }
            };
        }
    }
}
