// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Demo.Gpt2Console
{
    /// <summary>
    /// User-facing console demo: load GPT-2 Small, decode tokens with KV-cache,
    /// report allocations/token and tokens/sec.
    ///
    /// Example:
    ///   dotnet run -c Release --project Demo/Gpt2ConsoleDemo -- \
    ///     --model models/gpt2_small.bin \
    ///     --prompt "The future of software development is" \
    ///     --tokens 64
    ///
    /// Required files (alongside --model or via --vocab / --merges):
    ///   gpt2_small.bin   — Overfit binary weights (see Scripts/convert_gpt2.py)
    ///   vocab.json       — GPT-2 BPE vocab
    ///   merges.txt       — GPT-2 BPE merges
    ///
    /// Path resolution order:
    ///   1. Explicit CLI arg
    ///   2. $OVERFIT_MODEL_DIR/<filename>
    ///   3. ./models/<filename>
    /// </summary>
    internal static class Program
    {
        private static int Main(string[] args)
        {
            try
            {
                var opts = CliOptions.Parse(args);
                if (opts is null) { return 0; }  // --help was printed

                Run(opts);
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"error: {ex.Message}");
                return 1;
            }
        }

        private static void Run(CliOptions opts)
        {
            // ─── Resolve paths ─────────────────────────────────────────────────
            var modelPath = ResolvePath(opts.ModelPath, "gpt2_small.bin");
            var vocabPath = ResolvePath(opts.VocabPath, "vocab.json");
            var mergesPath = ResolvePath(opts.MergesPath, "merges.txt");

            var config = opts.Size switch
            {
                "Small" => Gpt2Config.Small,
                "Medium" => Gpt2Config.Medium,
                "Large" => Gpt2Config.Large,
                "XL" => Gpt2Config.XL,
                _ => throw new ArgumentException($"Unknown size: {opts.Size}")
            };

            Console.WriteLine($"GPT-2 {opts.Size}");
            Console.WriteLine($"  model:     {modelPath}");
            Console.WriteLine($"  vocab:     {vocabPath}");
            Console.WriteLine($"  merges:    {mergesPath}");
            Console.WriteLine($"  prompt:    \"{opts.Prompt}\"");
            Console.WriteLine($"  tokens:    {opts.MaxTokens}");

            // ─── Load model + tokenizer + engine in one call ───────────────────
            // Gpt2.Load throws FileNotFoundException with the missing path baked
            // in, so we don't need a separate existence-check step.
            using var gpt2 = Gpt2.Load(modelPath, vocabPath, mergesPath, config);

            var promptIds = gpt2.Tokenizer.Encode(opts.Prompt);
            Console.WriteLine($"  prompt tokens: {promptIds.Length}");

            using var session = gpt2.CreateSession();
            session.Reset(promptIds);

            Console.WriteLine("  KV-cache:  enabled");
            Console.WriteLine();

            // ─── Generation loop with split metrics ────────────────────────────
            // Two zones are measured separately so the headline claim
            // (0 B / generated token from KV-cache inference) doesn't get
            // polluted by .NET string allocations from tokenizer.DecodeToken
            // and Console.Write — those are unavoidable in any streaming demo
            // but live OUTSIDE the inference contract.
            //
            // - Inference zone:  only the GenerateNextToken() call is timed
            //                    and its per-iter managed allocation delta is
            //                    summed. This must report 0 B/token.
            // - Full loop zone:  outer stopwatch covers inference + decode +
            //                    Console.Write. Reported separately for
            //                    end-to-end user-visible cost.

            // Warm everything up so the first measured token isn't dominated
            // by codegen + JIT.
            var sampling = SamplingOptions.Greedy;
            _ = session.GenerateNextToken(in sampling);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var remaining = Math.Max(0, opts.MaxTokens - 1);
            long inferTicksSum = 0;
            long inferAllocSum = 0;

            var fullSw = Stopwatch.StartNew();
            Console.Write(opts.Prompt);

            for (var i = 0; i < remaining; i++)
            {
                // precise:false samples this thread's cumulative bytes only —
                // cheap, no cross-thread coordination, fine for per-iter use.
                var allocBefore = GC.GetTotalAllocatedBytes(precise: false);
                var tBefore = Stopwatch.GetTimestamp();

                var tokenId = session.GenerateNextToken(in sampling);

                inferTicksSum += Stopwatch.GetTimestamp() - tBefore;
                inferAllocSum += GC.GetTotalAllocatedBytes(precise: false) - allocBefore;

                Console.Write(gpt2.Tokenizer.DecodeToken(tokenId));
            }

            fullSw.Stop();

            Console.WriteLine();
            Console.WriteLine();

            // ─── Report ────────────────────────────────────────────────────────
            var generated = remaining;
            if (generated == 0)
            {
                Console.WriteLine("(no tokens generated — pass --tokens >= 2)");
                return;
            }

            var inferElapsedMs = inferTicksSum / (double)Stopwatch.Frequency * 1000.0;
            var inferTokensPerS = generated * 1000.0 / inferElapsedMs;
            var inferBytesPerTok = (double)inferAllocSum / generated;

            var fullElapsedMs = fullSw.Elapsed.TotalMilliseconds;
            var fullTokensPerS = generated / fullSw.Elapsed.TotalSeconds;
            // Full-loop alloc is dominated by string decode + Console.Write.
            // We don't measure it precisely here; see Gpt2TokensPerSecondBenchmark
            // for the BenchmarkDotNet-grade number.

            Console.WriteLine("--- Inference only (GenerateNextToken) ---");
            Console.WriteLine($"  Generated tokens:      {generated}");
            Console.WriteLine($"  Elapsed:               {inferElapsedMs:F1} ms");
            Console.WriteLine($"  Tokens/sec:            {inferTokensPerS:F1}");
            Console.WriteLine($"  Managed bytes / token: {inferBytesPerTok:F1}  (total: {inferAllocSum:N0} B)");
            Console.WriteLine();
            Console.WriteLine("--- Full demo loop (inference + decode + Console.Write) ---");
            Console.WriteLine($"  Elapsed:               {fullElapsedMs:F1} ms");
            Console.WriteLine($"  Tokens/sec:            {fullTokensPerS:F1}");
            Console.WriteLine("  (string + console alloc dominates; not part of the 0 B / token claim)");
        }

        private static string ResolvePath(string? cliValue, string fileName)
        {
            // 1. Explicit CLI argument wins.
            if (!string.IsNullOrWhiteSpace(cliValue))
            {
                return cliValue;
            }

            // 2. $OVERFIT_MODEL_DIR/<fileName>
            var envDir = Environment.GetEnvironmentVariable("OVERFIT_MODEL_DIR");
            if (!string.IsNullOrWhiteSpace(envDir))
            {
                var fromEnv = Path.Combine(envDir, fileName);
                if (File.Exists(fromEnv))
                {
                    return fromEnv;
                }
            }

            // 3. ./models/<fileName> (relative to working directory)
            return Path.Combine("models", fileName);
        }

        // File-existence checks live in Gpt2.Load now (throws FileNotFoundException
        // with the missing path baked in). No local EnsureExists helper needed.

        private sealed class CliOptions
        {
            public string? ModelPath { get; set; }
            public string? VocabPath { get; set; }
            public string? MergesPath { get; set; }
            public string Prompt { get; set; } = "The future of software development is";
            public int MaxTokens { get; set; } = 32;
            public string Size { get; set; } = "Small";

            public static CliOptions? Parse(string[] args)
            {
                var o = new CliOptions();
                for (var i = 0; i < args.Length; i++)
                {
                    switch (args[i])
                    {
                        case "-h":
                        case "--help":
                            PrintHelp();
                            return null;
                        case "--model": o.ModelPath = Take(args, ref i); break;
                        case "--vocab": o.VocabPath = Take(args, ref i); break;
                        case "--merges": o.MergesPath = Take(args, ref i); break;
                        case "--prompt": o.Prompt = Take(args, ref i); break;
                        case "--tokens": o.MaxTokens = int.Parse(Take(args, ref i)); break;
                        case "--size": o.Size = Take(args, ref i); break;
                        default:
                            throw new ArgumentException($"Unknown argument: {args[i]}");
                    }
                }

                if (o.MaxTokens <= 0)
                {
                    throw new ArgumentException("--tokens must be > 0.");
                }
                return o;
            }

            private static string Take(string[] args, ref int i)
            {
                if (i + 1 >= args.Length)
                {
                    throw new ArgumentException($"Missing value after {args[i]}.");
                }
                return args[++i];
            }

            private static void PrintHelp()
            {
                Console.WriteLine("Gpt2ConsoleDemo — pure C# GPT-2 inference demo");
                Console.WriteLine();
                Console.WriteLine("Usage: Gpt2ConsoleDemo [options]");
                Console.WriteLine();
                Console.WriteLine("  --model PATH     Overfit binary weights (default: ./models/gpt2_small.bin)");
                Console.WriteLine("  --vocab PATH     GPT-2 vocab.json       (default: ./models/vocab.json)");
                Console.WriteLine("  --merges PATH    GPT-2 merges.txt       (default: ./models/merges.txt)");
                Console.WriteLine("  --prompt TEXT    Prompt text            (default: \"The future of software development is\")");
                Console.WriteLine("  --tokens N       Total tokens to emit   (default: 32)");
                Console.WriteLine("  --size SIZE      Small|Medium|Large|XL  (default: Small)");
                Console.WriteLine("  -h, --help       Show this message");
                Console.WriteLine();
                Console.WriteLine("Env: OVERFIT_MODEL_DIR overrides ./models/ when a file isn't explicitly named.");
            }
        }
    }
}
