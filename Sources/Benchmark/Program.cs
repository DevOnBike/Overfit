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
        /// </summary>
        private static void Main(string[] args)
        {
            OverfitLicense.SuppressNotice = true;
            OverfitLicense.MessageSink = _ => { };

            BenchmarkSwitcher
                .FromAssembly(typeof(Program).Assembly)
                .Run(args);
        }
    }
}
