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
        /// System-wide lock name. <c>Global\</c> rather than <c>Local\</c> deliberately: the point is to catch
        /// a second run started from another terminal, another user session or a CI agent on the same box,
        /// which a session-scoped mutex would miss.
        /// </summary>
        private const string SingleRunMutexName = @"Global\DevOnBike.Overfit.Benchmarks";

        /// <summary>Exit code for "someone else is measuring" — distinct from a benchmark failure.</summary>
        private const int BusyExitCode = 2;
    }
}
