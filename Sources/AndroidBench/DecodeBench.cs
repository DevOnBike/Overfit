// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.AndroidBench
{
    /// <summary>
    /// UI-agnostic decode benchmark: loads a Q4_K_M GGUF, runs a correctness self-test (the SIMD path
    /// must yield the same tokens as the scalar oracle — they are bit-identical by construction), then
    /// A/B-measures decode tok/s with the fast path (NEON SDOT on arm64, AVX2 on x86) vs forced-scalar
    /// on the SAME device. Output goes through a <see cref="Action{T}"/> sink so a console
    /// <c>Main</c> or an Android <c>Activity</c> can both drive it.
    /// </summary>
    public static class DecodeBench
    {
        public static void Run(string modelPath, int genTokens, int repeats, int warmup, Action<string> log)
        {
            // Phone big.LITTLE diagnosis: the decode spin-pool PURE-hot-spins every core (incl. the slow
            // efficiency cores) — fine on a dedicated desktop, wasteful on a phone. Disable it and cap the
            // worker count to the big cluster. Must be set BEFORE any Overfit type loads (OverfitParallel
            // reads these at static-init). Read by env name to avoid pulling internals into this project.
            var poolEnv = Environment.GetEnvironmentVariable("OVERFIT_BENCH_POOL");
            var workersEnv = Environment.GetEnvironmentVariable("OVERFIT_BENCH_WORKERS");
            Environment.SetEnvironmentVariable("OVERFIT_DECODE_POOL", string.IsNullOrEmpty(poolEnv) ? "0" : poolEnv);
            Environment.SetEnvironmentVariable("OVERFIT_DECODE_WORKERS", string.IsNullOrEmpty(workersEnv) ? "4" : workersEnv);
            log($"config: OVERFIT_DECODE_POOL={Environment.GetEnvironmentVariable("OVERFIT_DECODE_POOL")} " +
                $"OVERFIT_DECODE_WORKERS={Environment.GetEnvironmentVariable("OVERFIT_DECODE_WORKERS")}");

            var fast = Dp.IsSupported ? "NEON(SDOT)" : Avx2.IsSupported ? "AVX2" : "scalar";

            log($"=== Overfit decode bench ===");
            log($"arch={RuntimeInformation.OSArchitecture} AdvSimd64={AdvSimd.Arm64.IsSupported} " +
                $"Dp(dotprod)={Dp.IsSupported} Avx2={Avx2.IsSupported} -> fast path = {fast}");
            log($"runtime: dynamicCode(JIT)={RuntimeFeature.IsDynamicCodeCompiled}  (False = AOT)");
            log($"model={modelPath}");
            log($"genTokens={genTokens} repeats={repeats} warmup={warmup}");

            log($"logical cores={Environment.ProcessorCount}");

            // quantize:false keeps the Q4_K weights Q4_K-resident so the FFN/attention decode runs through
            // Q4KDotKernel.MainDot — the kernel we NEON-ported. With quantize:true (the facade default) those
            // tensors are re-quantized to Q8_0 and dispatch to Q8DotKernel instead, leaving MainDot (and the
            // ForceScalar toggle) off the hot path — which makes the NEON-vs-scalar A/B a no-op.
            log("load: quantize=false (Q4_K-resident -> exercises Q4KDotKernel.MainDot / NEON)");
            using var client = OverfitClient.LoadGguf(modelPath, mmap: true, quantize: false);
            var tokenizer = client.Tokenizer;

            log($"RAM after load (RSS)={ReadVmRssMb()} MB  threads={ReadThreadCount()}");

            const string prompt = "The capital of France is";
            Span<int> promptBuffer = stackalloc int[64];
            var promptLength = tokenizer.Encode(prompt, promptBuffer);
            var promptIds = promptBuffer[..promptLength].ToArray();

            using var session = client.Engine.CreateSession(2048);
            var sampling = SamplingOptions.Greedy;

            // ── Correctness self-test: the fast path and the scalar oracle must produce identical tokens.
            // MainDotNeon/AVX2 are bit-identical to MainDotScalar, and greedy decode is deterministic, so
            // any divergence is a real kernel bug on this silicon — in which case the perf numbers are moot.
            var fastSequence = DecodeSequence(session, promptIds, genTokens, sampling, forceScalar: false);
            var scalarSequence = DecodeSequence(session, promptIds, genTokens, sampling, forceScalar: true);
            var identical = fastSequence.AsSpan().SequenceEqual(scalarSequence);

            log($"self-test {fast}==scalar over {genTokens} tokens: {(identical ? "PASS" : "FAIL")}");
            if (!identical)
            {
                log("  !! fast path diverged from scalar on this device — kernel bug; perf below is untrustworthy.");
            }

            log($"sample decode: {tokenizer.DecodeToString(fastSequence)}");

            // ── Perf A/B: best-of-repeats, interleaved per round so thermal/scheduler drift hits both sides.
            double bestFast = 0;
            double bestScalar = 0;
            for (var round = 0; round < repeats; round++)
            {
                var (fastTs, fastCores) = MeasureTokensPerSecond(session, promptIds, genTokens, warmup, sampling, forceScalar: false);
                var (scalarTs, scalarCores) = MeasureTokensPerSecond(session, promptIds, genTokens, warmup, sampling, forceScalar: true);
                bestFast = Math.Max(bestFast, fastTs);
                bestScalar = Math.Max(bestScalar, scalarTs);
                log($"  round {round + 1}/{repeats}: {fast}={fastTs:F2} tok/s ({fastCores:F1} cores)  " +
                    $"scalar={scalarTs:F2} tok/s ({scalarCores:F1} cores)  RSS={ReadVmRssMb()} MB");
            }

            Q4KDotKernel.ForceScalar = false;

            log($"--- result (best of {repeats}) ---");
            log($"{fast,-12}: {bestFast:F2} tok/s");
            log($"{"scalar",-12}: {bestScalar:F2} tok/s");
            log($"speedup     : {(bestScalar > 0 ? bestFast / bestScalar : 0):F2}x  ({fast} vs forced-scalar, same device)");

            // Component breakdown: where does single-token decode actually spend time on THIS device?
            // Settles whether the F32 lm-head, the Q4_K FFN, or attention dominates (i.e. which lever pays).
            log("--- decode profile (NEON, one pass) ---");
            DecodeProfiler.Enabled = true;
            DecodeProfiler.Reset();
            Q4KDotKernel.ForceScalar = false;
            session.Reset(promptIds);
            for (var i = 0; i < genTokens; i++)
            {
                session.GenerateNextToken(in sampling);
            }
            foreach (var line in DecodeProfiler.Report().Split('\n'))
            {
                var trimmed = line.TrimEnd();
                if (trimmed.Length > 0)
                {
                    log(trimmed);
                }
            }
            DecodeProfiler.Enabled = false;
        }

        private static int[] DecodeSequence(
            CachedLlamaSession session, int[] promptIds, int count, SamplingOptions sampling, bool forceScalar)
        {
            Q4KDotKernel.ForceScalar = forceScalar;
            session.Reset(promptIds);

            var output = new int[count];
            for (var i = 0; i < count; i++)
            {
                output[i] = session.GenerateNextToken(in sampling);
            }

            return output;
        }

        private static (double TokensPerSecond, double CoresBusy) MeasureTokensPerSecond(
            CachedLlamaSession session, int[] promptIds, int count, int warmup, SamplingOptions sampling, bool forceScalar)
        {
            Q4KDotKernel.ForceScalar = forceScalar;
            session.Reset(promptIds);

            for (var i = 0; i < warmup; i++)
            {
                session.GenerateNextToken(in sampling);
            }

            var jiffies0 = ReadCpuJiffies();
            var start = Stopwatch.GetTimestamp();
            for (var i = 0; i < count; i++)
            {
                session.GenerateNextToken(in sampling);
            }
            var seconds = (Stopwatch.GetTimestamp() - start) / (double)Stopwatch.Frequency;
            var jiffies1 = ReadCpuJiffies();

            var tokensPerSecond = seconds > 0 ? count / seconds : 0;
            // _SC_CLK_TCK is 100 on Android/Linux, so jiffies/100 = CPU-seconds; ÷ wall-seconds = cores busy.
            var coresBusy = jiffies0 >= 0 && jiffies1 >= 0 && seconds > 0
                ? (jiffies1 - jiffies0) / 100.0 / seconds
                : -1;

            return (tokensPerSecond, coresBusy);
        }

        private static long ReadVmRssMb()
        {
            try
            {
                foreach (var line in File.ReadLines("/proc/self/status"))
                {
                    if (line.StartsWith("VmRSS:", StringComparison.Ordinal))
                    {
                        var parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                        return long.Parse(parts[1]) / 1024;   // kB -> MB
                    }
                }
            }
            catch
            {
                // /proc absent (e.g. Windows dev box) — resource sampling is best-effort.
            }

            return -1;
        }

        private static int ReadThreadCount()
        {
            try
            {
                foreach (var line in File.ReadLines("/proc/self/status"))
                {
                    if (line.StartsWith("Threads:", StringComparison.Ordinal))
                    {
                        var parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                        return int.Parse(parts[1]);
                    }
                }
            }
            catch
            {
                // best-effort
            }

            return -1;
        }

        private static long ReadCpuJiffies()
        {
            try
            {
                var stat = File.ReadAllText("/proc/self/stat");
                var afterComm = stat.LastIndexOf(')');   // comm field can hold spaces/parens
                var fields = stat[(afterComm + 2)..].Split(' ', StringSplitOptions.RemoveEmptyEntries);
                // After comm: index 11 = utime, index 12 = stime (in clock ticks).
                return long.Parse(fields[11]) + long.Parse(fields[12]);
            }
            catch
            {
                return -1;
            }
        }
    }
}
