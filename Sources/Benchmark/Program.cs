// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Licensing;

namespace Benchmarks
{
    internal static class Program
    {
        /// <summary>
        /// Entry point uses <see cref="BenchmarkSwitcher"/> so the standard
        /// BenchmarkDotNet CLI works end-to-end:
        ///
        ///   dotnet run -c Release --project Sources/Benchmark --filter "*Gpt2Tokens*"
        ///   dotnet run -c Release --project Sources/Benchmark --filter "*"
        ///   dotnet run -c Release --project Sources/Benchmark               # interactive picker
        ///
        /// Headline benchmarks for the current GPT-2 showcase week:
        ///   *Gpt2TokensPerSecond*  — Legacy vs KV-cache vs Prefill-only, tokens/sec + alloc/op
        ///   *Gpt2KvCache*          — KV-cache memory and decode characteristics
        ///   *Gpt1Generation*       — GPT-1-scale end-to-end generation
        ///
        /// <para><b>One run at a time, enforced.</b> Two benchmark processes on one box do not produce two
        /// results — they produce two wrong ones, and nothing in the output says so. They compete for cores,
        /// L3, memory bandwidth and the same thermal budget, which is the entire set of things a measurement
        /// here is trying to hold still. Not hypothetical for this repository: the A/B discipline in
        /// CLAUDE.md exists because cross-process drift on this machine has already reached ~30%, and a
        /// concurrent run is that failure induced deliberately.</para>
        ///
        /// <para>The lock refuses rather than queues. Waiting would hide the collision behind a long pause,
        /// and "am I measuring or waiting?" is worth more than the convenience. BenchmarkDotNet's
        /// per-benchmark child processes are generated programs with their own entry point, so they never
        /// re-enter this method and cannot deadlock against the parent.</para>
        ///
        /// <para><b>Exit codes.</b> <c>0</c> at least one benchmark produced a measurement, or the caller
        /// asked an informational question (<c>--list</c>, <c>--info</c>, <c>--help</c>);
        /// <see cref="BusyExitCode"/> something else is measuring; <see cref="NothingRanExitCode"/> the host
        /// started and nothing was measured. The last one exists because BenchmarkDotNet reports a failed
        /// build as log text and returns normally — see <see cref="NothingRanExitCode"/>.</para>
        /// </summary>
        private static int Main(string[] args)
        {
            using var singleRun = new Mutex(initiallyOwned: false, SingleRunMutexName, out _);
            var acquired = false;

            try
            {
                try
                {
                    acquired = singleRun.WaitOne(TimeSpan.Zero);
                }
                catch (AbandonedMutexException)
                {
                    // A previous run died without releasing it. The lock is ours and nobody is measuring.
                    acquired = true;
                }

                if (!acquired)
                {
                    Console.Error.WriteLine("Another Overfit benchmark process is already running on this machine.");
                    Console.Error.WriteLine(
                        "Refusing to start: two concurrent runs share cores, cache, memory bandwidth and thermal "
                        + "headroom, so both sets of numbers would be wrong without saying so.");
                    Console.Error.WriteLine("Wait for it to finish, or stop it, then run again.");

                    return BusyExitCode;
                }

                if (MeasurementInProgress(out var lockPath))
                {
                    Console.Error.WriteLine($"A lab measurement is running ({lockPath} is held).");
                    Console.Error.WriteLine(
                        "Refusing to start: the cluster under measurement runs on this machine, so a "
                        + "benchmark's CPU load lands in its window as real cluster load. That has already "
                        + "happened once here — three CPU incidents appeared inside the window where builds "
                        + "were running, and they were indistinguishable from the cluster misbehaving.");
                    Console.Error.WriteLine("Wait for the run to finish, or stop the watcher, then run again.");

                    return BusyExitCode;
                }

                // Claim ownership of the machine for the build guard in Directory.Build.targets, which
                // refuses any build while this mutex is held. BenchmarkDotNet compiles a generated project
                // per job while this process holds it, and those compilers are children that inherit this
                // environment — so without the claim the guard would deadlock the benchmark against itself
                // rather than against a human. Ownership rather than exemption: a build carrying this pid
                // belongs to the run in progress.
                Environment.SetEnvironmentVariable(
                    MeasurementOwnerVariable,
                    Environment.ProcessId.ToString(System.Globalization.CultureInfo.InvariantCulture));

                OverfitLicense.SuppressNotice = true;
                OverfitLicense.MessageSink = _ => { };

                // The GGUF driver is answered here rather than through BenchmarkDotNet, and it still runs
                // under the machine mutex taken above — it is a measurement and must exclude the others.
                // BenchmarkDotNet is not involved at all: this mode exists so an external harness can
                // wall-clock this process and llama-bench's identically, using neither one's internal
                // timer, which is the only shape in which a cross-engine ratio means anything.
                if (Array.Exists(args, arg =>
                        string.Equals(arg, GgufBenchDriver.Switch, StringComparison.OrdinalIgnoreCase)))
                {
                    return GgufBenchDriver.Run(args);
                }

                // Our own switch is removed before BenchmarkDotNet sees the command line. It parses args
                // strictly, and an option it does not know is a REJECTED command line — which returns zero
                // summaries and looks exactly like "the filter matched nothing" (measured while closing
                // XC-47). Passing it through would turn an opt-in into a silent void run.
                var allowPartial = TakeAllowPartial(ref args);

                var summaries = BenchmarkSwitcher
                    .FromAssembly(typeof(Program).Assembly)
                    .Run(args)
                    .ToList();

                return ExitCodeFor(summaries, args, allowPartial);
            }
            finally
            {
                if (acquired)
                {
                    singleRun.ReleaseMutex();
                }
            }
        }

        /// <summary>
        /// System-wide lock name, <b>shared with the test suite</b> — see <c>Tests/MeasurementExclusion.cs</c>,
        /// which takes the same one.
        ///
        /// <para><c>Global\</c> rather than <c>Local\</c> deliberately: the point is to catch a second run
        /// started from another terminal, another user session or a CI agent on the same box, which a
        /// session-scoped mutex would miss.</para>
        ///
        /// <para><b>One name for both, because the collision is symmetric.</b> Two benchmarks corrupt each
        /// other; a suite starting mid-benchmark saturates thirty-two cores inside the sampling window; a
        /// benchmark starting mid-suite does the same in reverse. All three are the same machine being asked
        /// to hold still for two things at once, so they queue behind one lock rather than three mechanisms
        /// each aware of part of the problem.</para>
        ///
        /// <para><b>Refusing beats detecting.</b> The alternative — run both and mark the result suspect —
        /// spends the forty minutes first and reports afterwards.</para>
        /// </summary>
        private const string SingleRunMutexName = @"Global\DevOnBike.Overfit.MachineMeasurement";

        /// <summary>Exit code for "someone else is measuring" — distinct from a benchmark failure.</summary>
        private const int BusyExitCode = 2;

        /// <summary>
        /// Exit code for "the host started, but nothing was measured".
        ///
        /// <para><b>Why this exists.</b> BenchmarkDotNet reports a failed build as log text and then returns
        /// normally, so before 2026-08-14 a run whose build errored printed <c>// Build Error: …</c>,
        /// <c>executed benchmarks: 0</c> and then <b>exit code 0</b>. A script, a CI step or an agent keying
        /// on the exit code could not tell that from a real run — a green signal over an empty result, which
        /// is the worst shape a failure can take because nothing downstream looks again.</para>
        ///
        /// <para>Distinct from <see cref="BusyExitCode"/> on purpose: "someone else is measuring" is an
        /// expected outcome a caller may retry, while "nothing ran" means the caller's premise was wrong and
        /// retrying will produce the same nothing.</para>
        /// </summary>
        private const int NothingRanExitCode = 3;

        /// <summary>
        /// Turns the run's summaries into an exit code, and says on stderr <b>which</b> kind of nothing
        /// happened when nothing did.
        ///
        /// <para>Three outcomes are deliberately kept apart, because they call for three different actions:
        /// a filter that matched no benchmark is a mistake in the command line; an empty selection with no
        /// filter is a picker that was dismissed; and cases that were selected but produced no measurement
        /// is a build or validation failure whose detail is in the log above. They share
        /// <see cref="NothingRanExitCode"/> — a caller only needs to know the run is void — but a human
        /// reading stderr should not have to guess which one they hit.</para>
        /// </summary>
        /// <param name="summaries">Everything <see cref="BenchmarkSwitcher.Run"/> returned.</param>
        /// <param name="args">The command line, used only to tell the three outcomes apart.</param>
        /// <returns>0 if at least one benchmark produced a measurement, otherwise <see cref="NothingRanExitCode"/>.</returns>
        private static int ExitCodeFor(IReadOnlyList<Summary> summaries, string[] args, bool allowPartial)
        {
            // --help / --list / --info are answered by BenchmarkDotNet with no summaries at all, and they
            // are not failures: the caller asked a question and got an answer. Checked first so the
            // "nothing ran" code never fires on a successful query.
            if (IsInformationalInvocation(args))
            {
                return 0;
            }

            var selectedCases = 0;
            var measured = 0;

            foreach (var summary in summaries)
            {
                selectedCases += summary.BenchmarksCases.Length;

                // A report exists for a benchmark that failed to build or failed to run, and it carries no
                // measurements. Counting measurements rather than reports is what separates "it ran" from
                // "it was attempted".
                measured += summary.Reports.Count(report => report.AllMeasurements.Count > 0);
            }

            if (measured > 0 && measured == selectedCases)
            {
                return 0;
            }

            if (measured > 0)
            {
                return PartialExitCodeFor(summaries, selectedCases, measured, allowPartial);
            }

            if (selectedCases > 0)
            {
                Console.Error.WriteLine(
                    $"No benchmark produced a measurement: {selectedCases} case(s) were selected and none of "
                    + "them ran.");
                Console.Error.WriteLine(
                    "Look above for '// Build Error', a validation error or a crashed child process — that is "
                    + "where the run stopped. The results, if any were written, are from an earlier run.");

                return NothingRanExitCode;
            }

            Console.Error.WriteLine("No benchmark was selected, so this run measured nothing.");

            var filter = FilterArgument(args);

            if (filter is not null)
            {
                Console.Error.WriteLine(
                    $"If the filter '{filter}' matched nothing: BenchmarkDotNet matches the fully-qualified "
                    + "name, so a class name normally needs a star on both sides — "
                    + "--filter \"*SingleInferenceBenchmark*\". Use --list flat to see the names this "
                    + "assembly actually exposes.");
            }

            if (filter is null)
            {
                Console.Error.WriteLine(
                    "Pass --filter \"*\" to run everything, or a pattern to run a subset; --list flat prints "
                    + "the available names.");
            }

            // Measured 2026-08-14 while proving this exit path: a REJECTED command line produces exactly the
            // same empty result as a filter that matched nothing — `--cli <missing path>` prints "The
            // provided CliPath … does NOT exist" and returns zero summaries. Nothing in the return value
            // separates the two, so the message above says "if" rather than asserting the filter is at
            // fault, and this line names the other cause instead of leaving the reader to be misled.
            Console.Error.WriteLine(
                "If the log above reports a rejected or unknown option instead, that is the cause — "
                + "BenchmarkDotNet returns the same empty result for a command line it could not parse.");

            return NothingRanExitCode;
        }

        /// <summary>
        /// Exit code for "some of the selected cases were measured and some were not".
        ///
        /// <para><b>Why this is separate from <see cref="NothingRanExitCode"/>, and worse than it.</b>
        /// `XC-47` fixed the all-or-nothing case: zero measurements now exits 3. `XC-48` is the same lie in
        /// miniature — measured on 2026-08-14, <c>--runtimes net48 --job Dry</c> selected 8 cases, measured
        /// 4, failed to build 4, and exited <b>0</b>. The summary table <i>is</i> populated, so a reader has
        /// no reason to count its rows against what was selected, and every downstream consumer of that run
        /// treats a half-measurement as a whole one.</para>
        ///
        /// <para><b>Default refuse, explicit opt-in.</b> The host cannot reliably separate a legitimate skip
        /// (a runtime this machine does not have) from a defect (a generated project that failed to
        /// compile) — both arrive as a report with no measurements, and only the free-text error message
        /// differs. Rather than guess, the run fails and prints every unmeasured case with whatever reason
        /// BenchmarkDotNet gave, so a human decides. <c>--allow-partial</c> says "I know, some of these
        /// cannot run here" and returns 0 — after printing the same list, because an opted-in partial run
        /// still must not look complete.</para>
        /// </summary>
        private const int PartlyMeasuredExitCode = 4;

        /// <summary>
        /// Reports every selected case that produced no measurement, and decides the exit code.
        ///
        /// <para>Printed even when <paramref name="allowPartial"/> is set: the opt-in changes the exit code,
        /// never the visibility. A caller who suppresses the code still needs the list, because the whole
        /// defect being fixed here is a populated table that hides what is missing from it.</para>
        /// </summary>
        private static int PartialExitCodeFor(
            IReadOnlyList<Summary> summaries,
            int selectedCases,
            int measured,
            bool allowPartial)
        {
            var writer = allowPartial ? Console.Out : Console.Error;

            writer.WriteLine();
            writer.WriteLine(
                $"PARTIAL RUN: {measured} of {selectedCases} selected case(s) produced a measurement. "
                + $"{selectedCases - measured} did not, and are NOT in the table above.");

            foreach (var summary in summaries)
            {
                foreach (var benchmarkCase in summary.BenchmarksCases)
                {
                    var report = summary[benchmarkCase];

                    if (report is not null && report.AllMeasurements.Count > 0)
                    {
                        continue;
                    }

                    writer.WriteLine("  - " + benchmarkCase.DisplayInfo);
                    writer.WriteLine("      reason: " + ReasonFor(report));
                }

                foreach (var error in summary.ValidationErrors)
                {
                    writer.WriteLine(
                        $"  ! validation ({(error.IsCritical ? "critical" : "non-critical")}): {error.Message}");
                }
            }

            if (allowPartial)
            {
                writer.WriteLine(
                    "--allow-partial was passed, so this run exits 0. The cases listed above were still not "
                    + "measured; do not quote this run as covering them.");

                return 0;
            }

            writer.WriteLine(
                "This exits non-zero on purpose. A run that measured only part of what it selected is not a "
                + "result — a missing runtime and a generated project that failed to compile look identical "
                + "here, and only one of them is acceptable. Read the reasons above; if they are all "
                + "expected on this machine, re-run with --allow-partial.");

            return PartlyMeasuredExitCode;
        }

        /// <summary>
        /// Whatever BenchmarkDotNet recorded about a case that produced nothing — build error first, because
        /// that is the one that distinguishes a missing runtime from broken code, and it is free text.
        /// </summary>
        private static string ReasonFor(BenchmarkReport report)
        {
            if (report is null)
            {
                return "no report at all — the case was selected but never reached the toolchain.";
            }

            if (!report.BuildResult.IsBuildSuccess)
            {
                var message = report.BuildResult.ErrorMessage;

                return string.IsNullOrWhiteSpace(message)
                    ? "the build failed and BenchmarkDotNet recorded no message."
                    : "build failed — " + Condense(message);
            }

            if (!report.GenerateResult.IsGenerateSuccess)
            {
                return "the generated project could not be created.";
            }

            return "it built and ran but produced no measurement — look for a crashed child process above.";
        }

        /// <summary>One line out of a compiler's multi-line output, so the list stays readable.</summary>
        private static string Condense(string message)
        {
            var flattened = message.Replace('\r', ' ').Replace('\n', ' ').Trim();

            while (flattened.Contains("  ", StringComparison.Ordinal))
            {
                flattened = flattened.Replace("  ", " ", StringComparison.Ordinal);
            }

            return flattened.Length <= 400 ? flattened : flattened[..400] + " …";
        }

        /// <summary>
        /// Removes <c>--allow-partial</c> from the command line and reports whether it was there.
        ///
        /// <para>It must not reach BenchmarkDotNet. An unrecognised option is a rejected command line, which
        /// returns zero summaries and is indistinguishable from a filter that matched nothing — the exact
        /// confusion `XC-47` had to write a message around.</para>
        /// </summary>
        private static bool TakeAllowPartial(ref string[] args)
        {
            var kept = new List<string>(args.Length);
            var present = false;

            foreach (var arg in args)
            {
                if (string.Equals(arg, "--allow-partial", StringComparison.OrdinalIgnoreCase))
                {
                    present = true;

                    continue;
                }

                kept.Add(arg);
            }

            args = kept.ToArray();

            return present;
        }

        /// <summary>
        /// Whether the command line asks BenchmarkDotNet a question rather than asking it to measure.
        ///
        /// <para>These all return zero summaries by design, so without this they would be indistinguishable
        /// from a run that measured nothing. Prefix matching rather than equality because the switches take
        /// a value in both forms — <c>--list flat</c> and <c>--list=flat</c>.</para>
        /// </summary>
        private static bool IsInformationalInvocation(string[] args)
        {
            foreach (var arg in args)
            {
                if (arg.StartsWith("--list", StringComparison.OrdinalIgnoreCase)
                    || arg.StartsWith("--info", StringComparison.OrdinalIgnoreCase)
                    || arg.StartsWith("--help", StringComparison.OrdinalIgnoreCase)
                    || arg.StartsWith("--version", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(arg, "-h", StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// The filter pattern the caller passed, or <see langword="null"/> if they passed none. Used only to
        /// pick the right message; <c>-f</c> and <c>--filter</c> are BenchmarkDotNet's two spellings, and
        /// either may carry its value in the next argument or after an <c>=</c>.
        /// </summary>
        private static string FilterArgument(string[] args)
        {
            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];

                if (arg.StartsWith("--filter=", StringComparison.OrdinalIgnoreCase))
                {
                    return arg["--filter=".Length..];
                }

                if (!string.Equals(arg, "--filter", StringComparison.OrdinalIgnoreCase)
                    && !string.Equals(arg, "-f", StringComparison.Ordinal))
                {
                    continue;
                }

                // A trailing --filter with no value: report the switch itself rather than an empty string,
                // which would read as "matched nothing" when the real problem is a missing argument.
                return i + 1 < args.Length ? args[i + 1] : arg;
            }

            return null;
        }

        /// <summary>
        /// Environment variable naming the process that owns the machine for the duration of a run, read by
        /// the <c>OverfitBuildExclusionCheck</c> target in <c>Directory.Build.targets</c>.
        ///
        /// <para>It exists so the build guard can tell "a human started a build during a benchmark" from
        /// "BenchmarkDotNet is compiling its own generated project", which are the same event to a mutex
        /// probe and opposite events to a human.</para>
        /// </summary>
        private const string MeasurementOwnerVariable = "OVERFIT_MEASUREMENT_OWNER";

        /// <summary>
        /// Whether a lab measurement holds its watcher lock.
        ///
        /// <para><b>A second kind of collision, and the one this repository actually suffered.</b> The mutex
        /// above stops two benchmarks competing with each other. It says nothing about a benchmark competing
        /// with the <i>cluster being measured</i>, which on this machine runs in Docker on the same cores —
        /// so a benchmark's load arrives inside the measurement window as if the cluster had produced it. On
        /// 2026-08-02 that produced three CPU incidents in a false-positive count, and nothing in either
        /// output said the two were related.</para>
        ///
        /// <para>The lock is a real OS-held file lock rather than a marker file, so it disappears when the
        /// watcher dies however it dies — a crashed run leaves nothing stale to clean up. Absent file, or a
        /// file nobody holds, means nobody is measuring.</para>
        /// </summary>
        private static bool MeasurementInProgress(out string lockPath)
        {
            // Walked up to the solution rather than counted in "..": the number of levels between the
            // binary and the repository root is a property of the output layout, and it changes without
            // anyone noticing that this check quietly stopped finding the file.
            lockPath = string.Empty;

            var directory = new DirectoryInfo(AppContext.BaseDirectory);

            while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Overfit.sln")))
            {
                directory = directory.Parent;
            }

            if (directory is null)
            {
                return false;
            }

            var full = Path.Combine(directory.FullName, "Tests", "bin", "fp-run.lock");
            lockPath = full;

            if (!File.Exists(full))
            {
                return false;
            }

            try
            {
                // Opening with no sharing succeeds only if nothing else holds it.
                using var probe = new FileStream(full, FileMode.Open, FileAccess.ReadWrite, FileShare.None);

                return false;
            }
            catch (IOException)
            {
                return true;
            }
            catch (UnauthorizedAccessException)
            {
                // Cannot tell. Refusing on "cannot tell" would block benchmarking on a permissions quirk;
                // the mutex above still guards the collision this class was originally written for.
                return false;
            }
        }
    }
}
