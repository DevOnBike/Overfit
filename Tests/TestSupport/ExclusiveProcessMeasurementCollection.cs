// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// The collection for tests that assert on a PROCESS-WIDE quantity — CPU time, working set, thread
    /// count — and therefore cannot share the process with anything else while they measure.
    ///
    /// <para><b>Putting a test in here serialises it against the ENTIRE suite</b>, not against a sibling
    /// class: xUnit runs collections in parallel and <c>DisableParallelization</c> takes this one out of
    /// that pool, so nothing else executes while a member of it runs. That is the cost — the suite loses
    /// the wall-clock of every test in here — and it is why membership needs a reason of the form "my
    /// instrument reads the whole process", not "this test is flaky".</para>
    ///
    /// <para><b>Why it exists (<c>XC-54</c>, 2026-08-15).</b>
    /// <c>DecodePoolIdleBurnTests.Pool_Parks_WhenIdle</c> asserts that the decode spin-pool burns under one
    /// effective core at idle, computed from <see cref="System.Diagnostics.Process.TotalProcessorTime"/> —
    /// the whole process. With collections running in parallel it measured <b>14.57 effective cores on a
    /// 32-logical-core box</b> and passed only in isolation: every other test executing in its 3-second
    /// window was counted as pool spin. It was not flaky, it was measuring the wrong subject.</para>
    ///
    /// <para><b>What this does and does not buy.</b> It gives the measurement a quiet window, which makes
    /// process CPU a fair proxy for the subject's CPU. It does <b>not</b> make the instrument able to tell
    /// the subject from the environment — a leaked pool from an earlier test, a GC storm, or a background
    /// thread still lands in the same counter. A member of this collection must still carry its own canary
    /// (measure the same quantity before the subject is woken) and report "could not measure" rather than
    /// pass or fail when the baseline is already busy.</para>
    /// </summary>
    [CollectionDefinition(Name, DisableParallelization = true)]
    public sealed class ExclusiveProcessMeasurementCollection
    {
        /// <summary>The name to put on <c>[Collection(...)]</c> at the test class.</summary>
        public const string Name = "exclusive-process-measurement";
    }
}
