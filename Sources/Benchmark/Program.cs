// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Running;
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

                OverfitLicense.SuppressNotice = true;
                OverfitLicense.MessageSink = _ => { };

                BenchmarkSwitcher
                    .FromAssembly(typeof(Program).Assembly)
                    .Run(args);

                return 0;
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
