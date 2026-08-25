// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

namespace Benchmarks.Helpers
{
    /// <summary>
    /// <c>llama-bench</c>'s command line and output shape, driving Overfit, so one harness can measure both
    /// engines through one parser.
    ///
    /// <code>
    /// Benchmarks.exe --gguf-bench -m C:\qwen3b\qwen.q4km.gguf -t 16 -p 512 -n 0 -r 3 -o json
    /// </code>
    ///
    /// <para><b>Why this exists next to the BenchmarkDotNet class rather than instead of it.</b>
    /// <see cref="GgufEndToEndThroughputBenchmark"/> is the artefact <c>XC-76</c> asks for — a benchmark in
    /// the repository that loads a real GGUF and measures end-to-end decode — and it reports through
    /// BenchmarkDotNet's own statistics. This driver exists because the comparison against llama.cpp must be
    /// measured by <b>one</b> instrument applied to both sides. On 2026-08-20 this project compared itself
    /// with another engine using that engine's mature CLI on one side and a harness written that morning on
    /// the other, and every suspicious number came through the new half. The fix found then was to take the
    /// intersection of what both sides already expose; <c>Scripts/gguf_bench.py</c> wall-clocks this process
    /// and llama-bench's identically and uses neither one's internal timer for the comparison.</para>
    ///
    /// <para><b>Two thread counts are reported, not one.</b> <c>-t</c> sets
    /// <c>OVERFIT_PARALLEL_WORKERS</c>-equivalent general parallelism, and the decode dispatch caps itself
    /// separately — so <c>n_threads</c> and <c>n_threads_decode</c> are both in the output and they normally
    /// differ. llama.cpp has one number here and Overfit has two; printing only the requested one would state
    /// a configuration the run did not have.</para>
    ///
    /// <para><b>The raw per-repetition timings are printed, not just their mean.</b> They are what catches a
    /// repetition that did no work — a cache that made repetitions 2..r free would show as a mean that looks
    /// merely fast, and as a sample array that is obviously wrong. <see cref="SampleSpreadWarningRatio"/>
    /// makes the check explicit rather than leaving it to whoever reads the array.</para>
    /// </summary>
    internal static class GgufBenchDriver
    {
        /// <summary>The switch that routes away from BenchmarkDotNet, stripped before it sees the args.</summary>
        public const string Switch = "--gguf-bench";

        /// <summary>
        /// Ratio of slowest to fastest repetition above which the run is called out on stderr.
        ///
        /// <para>Set loose on purpose. It is not a noise threshold — it exists to catch a repetition that
        /// did not run the work at all, which is a factor of hundreds, not of two. A tight value here would
        /// fire on ordinary thermal spread and teach the reader to ignore the line.</para>
        /// </summary>
        private const double SampleSpreadWarningRatio = 3.0;

        public static int Run(string[] args)
        {
            var model = Value(args, "-m") ?? Value(args, "--model");
            var threads = Integer(args, "-t", Environment.ProcessorCount);
            var prompt = Integer(args, "-p", 512);
            var generate = Integer(args, "-n", 0);
            var repetitions = Integer(args, "-r", 3);
            var warmup = !Has(args, "--no-warmup");

            if (string.IsNullOrWhiteSpace(model))
            {
                Console.Error.WriteLine($"{Switch}: -m <model.gguf> is required.");

                return 2;
            }

            if (!File.Exists(model))
            {
                Console.Error.WriteLine($"{Switch}: model not found: {model}");

                return 2;
            }

            if (prompt > 0 && generate > 0)
            {
                // llama-bench splits `-p N -n M` into two independent tests. Refusing rather than silently
                // measuring a mixed run keeps this driver's one row meaning one thing.
                Console.Error.WriteLine(
                    $"{Switch}: exactly one of -p and -n must be non-zero; a mixed run is a third quantity.");

                return 2;
            }

            if (prompt <= 0 && generate <= 0)
            {
                Console.Error.WriteLine($"{Switch}: one of -p / -n must be greater than zero.");

                return 2;
            }

            // Set before anything touches the engine: OverfitParallel resolves its worker count once, in a
            // static initializer. A caller that sets this after the first dispatch changes nothing and gets
            // no warning, which is how an A/B ends up running the same arm twice.
            Environment.SetEnvironmentVariable(
                DevOnBike.Overfit.Runtime.OverfitEnvironment.ParallelWorkers,
                threads.ToString(CultureInfo.InvariantCulture));

            var decoding = generate > 0;
            var tokens = decoding ? generate : prompt;

            using var probe = new GgufThroughputProbe(model, Math.Max(prompt, 1), generate);
            var samples = new long[repetitions];

            if (warmup)
            {
                RunOne(probe, decoding);
            }

            for (var i = 0; i < repetitions; i++)
            {
                samples[i] = RunOne(probe, decoding);
            }

            Report(model, threads, prompt, generate, tokens, samples);

            return 0;
        }

        /// <summary>One repetition: untimed reset, timed body, elapsed nanoseconds.</summary>
        private static long RunOne(GgufThroughputProbe probe, bool decoding)
        {
            if (decoding)
            {
                probe.ResetForDecode();
            }

            if (!decoding)
            {
                probe.ResetForPrefill();
            }

            var start = Stopwatch.GetTimestamp();

            if (decoding)
            {
                probe.RunDecode();
            }

            if (!decoding)
            {
                probe.RunPrefill();
            }

            var ticks = Stopwatch.GetTimestamp() - start;

            return (long)(ticks * (1_000_000_000.0 / Stopwatch.Frequency));
        }

        /// <summary>
        /// llama-bench's field names, so <c>Scripts/gguf_bench.py</c> reads one shape from both engines.
        /// <c>avg_ts</c> is the mean of the per-repetition rates — the same definition llama-bench uses,
        /// which is not the rate at the mean time and would differ from it if the samples were spread.
        /// </summary>
        private static void Report(
            string model, int threads, int prompt, int generate, int tokens, long[] samples)
        {
            var rates = new double[samples.Length];

            for (var i = 0; i < samples.Length; i++)
            {
                rates[i] = tokens / (samples[i] / 1_000_000_000.0);
            }

            var meanNs = Mean(samples);
            var meanTs = Mean(rates);
            var slowest = samples[0];
            var fastest = samples[0];

            foreach (var sample in samples)
            {
                slowest = Math.Max(slowest, sample);
                fastest = Math.Min(fastest, sample);
            }

            if (fastest > 0 && slowest / (double)fastest > SampleSpreadWarningRatio)
            {
                Console.Error.WriteLine(
                    $"{Switch}: repetition spread {slowest / (double)fastest:F1}x — the fast repetitions may "
                    + "not have done the work. Read samples_ns before using this row.");
            }

            var text = new StringBuilder();
            text.Append("[\n  {\n");
            text.Append("    \"engine\": \"overfit\",\n");

            // The repository's own commit is NOT reported here, deliberately: Scripts/provenance.py owns
            // that record, knows the dirty list and the assembly hashes as well as HEAD, and a second
            // half-answer printed beside it would be the copy that goes stale.
            text.Append(CultureInfo.InvariantCulture,
                $"    \"runtime\": \"{Escape(RuntimeInformation.FrameworkDescription)}\",\n");
            text.Append(CultureInfo.InvariantCulture,
                $"    \"cpu_info\": \"{Escape(Cpu())}\",\n");
            text.Append(CultureInfo.InvariantCulture,
                $"    \"model_filename\": \"{Escape(model)}\",\n");
            text.Append(CultureInfo.InvariantCulture,
                $"    \"model_size\": {new FileInfo(model).Length},\n");

            // The offline-repacked sidecar, declared because the loader consumes it with no switch and no
            // announcement (GgufLlamaLoader.TryOpenSidecar, unconditional) and llama.cpp has no equivalent.
            // A row that does not say whether it was there is a row nobody can place.
            //
            // PRESENCE IS NOT PROOF OF USE and this field must not be read as proof: a sidecar whose
            // tensors do not match is silently ignored (AttachPrepacked returns without attaching), and a
            // corrupt one is swallowed by TryOpenSidecar. The only evidence that it was used is a measured
            // difference between a run with it and a run without it.
            var sidecar = model + ".repack";
            var present = File.Exists(sidecar);
            text.Append(CultureInfo.InvariantCulture,
                $"    \"repack_sidecar_present\": {(present ? "true" : "false")},\n");
            text.Append(CultureInfo.InvariantCulture,
                $"    \"repack_sidecar_bytes\": {(present ? new FileInfo(sidecar).Length : 0L)},\n");
            text.Append(CultureInfo.InvariantCulture, $"    \"n_threads\": {GgufThroughputProbe.WorkerCount},\n");
            text.Append(CultureInfo.InvariantCulture,
                $"    \"n_threads_requested\": {threads},\n");
            text.Append(CultureInfo.InvariantCulture,
                $"    \"n_threads_decode\": {GgufThroughputProbe.DecodeWorkerCount},\n");
            text.Append(CultureInfo.InvariantCulture, $"    \"n_prompt\": {prompt},\n");
            text.Append(CultureInfo.InvariantCulture, $"    \"n_gen\": {generate},\n");
            text.Append(CultureInfo.InvariantCulture, $"    \"n_depth\": 0,\n");
            text.Append(CultureInfo.InvariantCulture, $"    \"avg_ns\": {meanNs:F0},\n");
            text.Append(CultureInfo.InvariantCulture, $"    \"stddev_ns\": {StandardDeviation(samples):F0},\n");
            text.Append(CultureInfo.InvariantCulture, $"    \"avg_ts\": {meanTs:F6},\n");
            text.Append(CultureInfo.InvariantCulture, $"    \"stddev_ts\": {StandardDeviation(rates):F6},\n");
            text.Append(CultureInfo.InvariantCulture, $"    \"samples_ns\": [ {Join(samples)} ],\n");
            text.Append(CultureInfo.InvariantCulture, $"    \"samples_ts\": [ {Join(rates)} ]\n");
            text.Append("  }\n]\n");

            Console.Out.Write(text.ToString());
        }

        /// <summary>
        /// A CPU description without WMI or a native call. <c>PROCESSOR_IDENTIFIER</c> is family/model
        /// rather than a marketing name, so it does not read the same as llama-bench's string; the harness
        /// records both and a reader compares them by hand rather than by string equality.
        /// </summary>
        private static string Cpu()
        {
            var identifier = Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER");

            return string.IsNullOrWhiteSpace(identifier)
                ? RuntimeInformation.ProcessArchitecture.ToString()
                : identifier;
        }

        private static double Mean(long[] values)
        {
            var total = 0.0;

            foreach (var value in values)
            {
                total += value;
            }

            return total / values.Length;
        }

        private static double Mean(double[] values)
        {
            var total = 0.0;

            foreach (var value in values)
            {
                total += value;
            }

            return total / values.Length;
        }

        private static double StandardDeviation(long[] values)
        {
            var doubles = new double[values.Length];

            for (var i = 0; i < values.Length; i++)
            {
                doubles[i] = values[i];
            }

            return StandardDeviation(doubles);
        }

        /// <summary>Sample standard deviation (n-1), which is what llama-bench reports.</summary>
        private static double StandardDeviation(double[] values)
        {
            if (values.Length < 2)
            {
                return 0.0;
            }

            var mean = Mean(values);
            var total = 0.0;

            foreach (var value in values)
            {
                total += (value - mean) * (value - mean);
            }

            return Math.Sqrt(total / (values.Length - 1));
        }

        private static string Join(long[] values)
        {
            var text = new StringBuilder();

            for (var i = 0; i < values.Length; i++)
            {
                if (i > 0)
                {
                    text.Append(", ");
                }

                text.Append(CultureInfo.InvariantCulture, $"{values[i]}");
            }

            return text.ToString();
        }

        private static string Join(double[] values)
        {
            var text = new StringBuilder();

            for (var i = 0; i < values.Length; i++)
            {
                if (i > 0)
                {
                    text.Append(", ");
                }

                text.Append(CultureInfo.InvariantCulture, $"{values[i]:F4}");
            }

            return text.ToString();
        }

        private static string Escape(string value)
        {
            return value.Replace("\\", "\\\\", StringComparison.Ordinal)
                .Replace("\"", "\\\"", StringComparison.Ordinal);
        }

        private static bool Has(string[] args, string name)
        {
            foreach (var arg in args)
            {
                if (string.Equals(arg, name, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }

        private static string? Value(string[] args, string name)
        {
            for (var i = 0; i < args.Length - 1; i++)
            {
                if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                {
                    return args[i + 1];
                }
            }

            return null;
        }

        private static int Integer(string[] args, string name, int fallback)
        {
            var raw = Value(args, name);

            return raw is not null && int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture,
                out var parsed)
                ? parsed
                : fallback;
        }
    }
}
