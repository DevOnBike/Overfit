// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Text;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// The refusal in <see cref="MeasurementExclusion"/> writes itself down, because it cannot say anything a
    /// human will read.
    ///
    /// <para><b>Why a file.</b> <c>XC-17</c> measured both ways out: through <c>dotnet test</c> the VSTest
    /// bridge launches this assembly's executable as a child and expects JSON on its stdout, so
    /// <c>Environment.Exit(2)</c> — and a thrown exception equally — surfaces as
    /// <c>Test process did not return valid JSON (non-object)</c> plus "No test is available", with exit code
    /// 1. The sentence written for the operator never reaches them, and the visible symptom points at the test
    /// discoverer instead of at a held lock.</para>
    ///
    /// <para><b>What that cost, once, measurably.</b> On 2026-08-13 a killed <c>[LongFact]</c> run left
    /// <c>DevOnBike.Overfit.Tests.exe</c> alive as an orphan — xUnit v3 makes the test project an executable,
    /// so it is a grandchild that killing vstest and testhost does not reach — still holding the mutex. Every
    /// later run refused, exactly as designed, and the diagnosis took a process-ancestry walk, a mutex probe
    /// and a command-line search over about an hour. Every fact needed to answer it in one read is in the text
    /// this class composes.</para>
    ///
    /// <para><b>The one sentence that does the work</b>: exit 2 means something <b>alive</b> holds the lock,
    /// never that something died holding it. A holder that dies is reported as
    /// <see cref="AbandonedMutexException"/>, which <see cref="MeasurementExclusion"/> already catches and
    /// treats as "the lock is ours".</para>
    ///
    /// <para>Nothing here may throw. A guard whose diagnostics fail is worse than one with none, because the
    /// failure arrives on top of the refusal it was supposed to explain.</para>
    /// </summary>
    internal static class MeasurementRefusalMarker
    {
        /// <summary>
        /// Name of the marker, written under <c>Tests/bin</c> — a directory a test run already writes to and
        /// git already ignores.
        /// </summary>
        internal const string FileName = "measurement-refusal.txt";

        /// <summary>
        /// Process name fragments searched when listing possible holders.
        ///
        /// <para><b>By name pattern, never by process class</b>, and this is measured rather than stylistic:
        /// on 2026-08-13 three separate filters built on <c>testhost|vstest|Benchmark</c> found nothing while
        /// the orphan sat in plain view, because the thing holding the lock was named after the test
        /// assembly. <c>dotnet</c> is deliberately absent — it matches every SDK process on the box and would
        /// bury the one line that matters.</para>
        /// </summary>
        private static readonly string[] NameFragments = ["Overfit", "Tests", "Benchmark", "testhost", "vstest"];

        /// <summary>
        /// Writes the marker for a refusal happening now, and returns the path written, or
        /// <see langword="null"/> if it could not be written.
        /// </summary>
        /// <param name="mutexName">The lock that was already held.</param>
        internal static string? Write(string mutexName)
        {
            var content = Compose(
                DateTimeOffset.Now,
                mutexName,
                Environment.ProcessId,
                ProcessNameOrUnknown(),
                CommandLineOrUnknown(),
                LiveCandidates());

            return TryWrite(ResolveDirectory(), content);
        }

        /// <summary>
        /// Builds the marker text. Pure, and takes every varying input as a parameter, so the text a human
        /// will read under a real refusal is the same text a test asserts on.
        /// </summary>
        /// <param name="when">Timestamp of the refusal.</param>
        /// <param name="mutexName">The lock that was already held.</param>
        /// <param name="processId">Process id of the run that was refused.</param>
        /// <param name="processName">Process name of the run that was refused.</param>
        /// <param name="commandLine">
        /// Command line of the run that was refused. It is here to answer "is this record mine?": the marker
        /// keeps only the latest refusal, and a reader coming back to it an hour later needs to tell their own
        /// blocked run from a later one.
        /// </param>
        /// <param name="candidates">Live processes matching <see cref="NameFragments"/>, one per line.</param>
        internal static string Compose(
            DateTimeOffset when,
            string mutexName,
            int processId,
            string processName,
            string commandLine,
            IReadOnlyList<string> candidates)
        {
            var text = new StringBuilder();

            text.AppendLine("Overfit measurement exclusion: THE TEST SUITE REFUSED TO START.");
            text.AppendLine();
            text.AppendLine($"When   : {when:yyyy-MM-dd HH:mm:ss.fff zzz}  ({when.ToUniversalTime():yyyy-MM-dd HH:mm:ss} UTC)");
            text.AppendLine($"Mutex  : {mutexName}");
            text.AppendLine($"Refused: pid {processId} {processName}");
            text.AppendLine($"         {commandLine}");
            text.AppendLine();
            text.AppendLine("WHAT THIS MEANS");
            text.AppendLine("  The machine-wide measurement lock was already held, so the run exited with code 2");
            text.AppendLine("  without executing a single test. A benchmark and a test suite may not share this box:");
            text.AppendLine("  thirty-two cores of test load inside a sampling window produces a wrong number that");
            text.AppendLine("  looks like a measurement.");
            text.AppendLine();
            text.AppendLine("  SOMETHING ALIVE HOLDS THE LOCK. This is never a process that died holding it: a dead");
            text.AppendLine("  owner surfaces as AbandonedMutexException, which the guard catches and treats as");
            text.AppendLine("  'nobody is measuring, the lock is ours'. So do not go looking for stale state to clean");
            text.AppendLine("  up — look for a running process.");
            text.AppendLine();
            text.AppendLine("HOW TO FIND THE HOLDER (Windows)");
            text.AppendLine("  Get-CimInstance Win32_Process |");
            text.AppendLine("    Where-Object { $_.Name -match 'Overfit|Tests' } |");
            text.AppendLine("    Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine");
            text.AppendLine();
            text.AppendLine("  Search by NAME PATTERN, not by process class. Measured 2026-08-13: three separate");
            text.AppendLine("  filters built on 'testhost|vstest|Benchmark' found nothing while the orphan sat in");
            text.AppendLine("  plain view, because it is named after the test assembly.");
            text.AppendLine();
            text.AppendLine("  A healthy run is a MATCHED PAIR — DevOnBike.Overfit.Tests.exe beside a testhost.exe");
            text.AppendLine("  with the same start time. The orphan signature is the test executable ALIVE WITH NO");
            text.AppendLine("  testhost next to it: xUnit v3 makes the test project an executable, so it is a");
            text.AppendLine("  grandchild that killing vstest and testhost leaves running, still holding this mutex.");
            text.AppendLine("  Kill the process TREE, not the runner.");
            text.AppendLine();
            text.AppendLine("CANDIDATE PROCESSES, AS SEEN AT THE MOMENT OF REFUSAL");

            if (candidates.Count == 0)
            {
                text.AppendLine("  (none matched — the holder is named something this list does not cover, or the");
                text.AppendLine("   process list could not be read)");
            }

            foreach (var candidate in candidates)
            {
                text.AppendLine($"  {candidate}");
            }

            text.AppendLine();
            text.AppendLine("WHY THE CONSOLE DID NOT TELL YOU THIS (XC-17)");
            text.AppendLine("  Through `dotnet test` the VSTest bridge runs this assembly's executable as a child and");
            text.AppendLine("  expects JSON on its stdout, so the refusal lands inside the bridge's first probe. The");
            text.AppendLine("  operator sees 'Test process did not return valid JSON (non-object)' and 'No test is");
            text.AppendLine("  available', with exit code 1 rather than 2. Run the executable directly to see the");
            text.AppendLine("  real message.");
            text.AppendLine();
            text.AppendLine("ESCAPE HATCH");
            text.AppendLine("  OVERFIT_ALLOW_CONCURRENT_MEASUREMENT=1 switches the exclusion off for one run. Use it");
            text.AppendLine("  when you know the two are not sharing a box; it is not a fix for an orphan.");
            text.AppendLine();
            text.AppendLine("Only the most recent refusal is kept. Check the timestamp above before acting on it.");
            text.AppendLine("A record whose command line contains '-list classes' is this guard's own end-to-end");
            text.AppendLine("test refusing a child process on purpose, and is not an incident.");

            return text.ToString();
        }

        /// <summary>
        /// Live processes whose name contains one of <see cref="NameFragments"/>, formatted one per line as
        /// <c>pid NNN name (started ...)</c>.
        ///
        /// <para>The orphan signature is annotated here rather than left to the reader: a process named after
        /// the test assembly with no <c>testhost</c> anywhere on the box is the 2026-08-13 shape exactly.</para>
        /// </summary>
        internal static IReadOnlyList<string> LiveCandidates()
        {
            var lines = new List<string>();

            try
            {
                var processes = Process.GetProcesses();
                var testHostSeen = false;

                foreach (var process in processes)
                {
                    if (process.ProcessName.Contains("testhost", StringComparison.OrdinalIgnoreCase))
                    {
                        testHostSeen = true;
                    }
                }

                foreach (var process in processes)
                {
                    if (!Matches(process.ProcessName))
                    {
                        continue;
                    }

                    lines.Add(Describe(process, testHostSeen));
                }

                foreach (var process in processes)
                {
                    process.Dispose();
                }
            }
            catch (InvalidOperationException)
            {
                // The process list moved under us. Nothing to report and nothing worth failing over — the
                // rest of the marker still answers the question.
            }

            return lines;
        }

        /// <summary>
        /// The directory the marker goes in: <c>Tests/bin</c> beside the solution, falling back to the output
        /// directory when the solution is not on disk (a published or copied test binary).
        ///
        /// <para>Walked up to <c>Overfit.sln</c> rather than counted in <c>..</c> segments, the same way
        /// <c>Sources/Benchmark/Program.cs</c> finds <c>fp-run.lock</c>: the number of levels between the
        /// binary and the repository root is a property of the output layout, and it changes without anyone
        /// noticing that a path quietly stopped resolving.</para>
        /// </summary>
        internal static string ResolveDirectory()
        {
            var directory = new DirectoryInfo(AppContext.BaseDirectory);

            // BOUND: strictly ascending a finite path; DirectoryInfo.Parent is null at the volume root.
            while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Overfit.sln")))
            {
                directory = directory.Parent;
            }

            if (directory is null)
            {
                return AppContext.BaseDirectory;
            }

            return Path.Combine(directory.FullName, "Tests", "bin");
        }

        /// <summary>
        /// Writes <paramref name="content"/> into <paramref name="directory"/>, returning the full path, or
        /// <see langword="null"/> if anything at all went wrong.
        /// </summary>
        /// <param name="directory">Target directory; created if missing.</param>
        /// <param name="content">Marker text.</param>
        internal static string? TryWrite(string directory, string content)
        {
            try
            {
                Directory.CreateDirectory(directory);

                var path = Path.Combine(directory, FileName);

                File.WriteAllText(path, content);

                return path;
            }
            catch (IOException)
            {
                return null;
            }
            catch (UnauthorizedAccessException)
            {
                return null;
            }
            catch (NotSupportedException)
            {
                return null;
            }
        }

        private static bool Matches(string processName)
        {
            foreach (var fragment in NameFragments)
            {
                if (processName.Contains(fragment, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }

        private static string Describe(Process process, bool testHostSeen)
        {
            var started = "start time unavailable";

            try
            {
                started = $"started {process.StartTime:yyyy-MM-dd HH:mm:ss}";
            }
            catch (Exception exception) when (exception is InvalidOperationException or SystemException)
            {
                // Win32Exception for a process this one may not open, InvalidOperationException if it exited
                // between the enumeration and here. Either way the pid and the name are the useful part.
            }

            var line = $"pid {process.Id,-8} {process.ProcessName}  ({started})";

            if (!testHostSeen && process.ProcessName.Contains("DevOnBike.Overfit.Tests", StringComparison.OrdinalIgnoreCase))
            {
                line += "  <- ORPHAN SIGNATURE: no testhost process on this box";
            }

            return line;
        }

        private static string ProcessNameOrUnknown()
        {
            try
            {
                using var current = Process.GetCurrentProcess();

                return current.ProcessName;
            }
            catch (InvalidOperationException)
            {
                return "(process name unavailable)";
            }
        }

        private static string CommandLineOrUnknown()
        {
            try
            {
                return Environment.CommandLine;
            }
            catch (NotSupportedException)
            {
                return "(command line unavailable)";
            }
        }
    }
}
