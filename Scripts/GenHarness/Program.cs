// Raw GGUF completion, timed the way dotLLM's `run --json` times it, so the two are comparable.
//
// WHY THIS EXISTS RATHER THAN `overfit chat`. Our CLI's chat command is interactive only — no --prompt, no
// --max-tokens, no bounded run — and both `client.Send` and `client.Complete` wrap the prompt in the model's
// CHAT TEMPLATE. dotLLM's `run` is a raw text completion. Comparing the two would compare different prompts,
// different token counts and different outputs, and the text-parity check would become impossible. So this
// goes one level below the chat layer, onto CachedLlamaSession — the same production runtime `chat` uses,
// just without the template.
//
// WHAT IT REPORTS AND WHY IN THESE PARTS. dotLLM splits load / prefill / decode / sampling; our
// GenerationStats does not split prefill from decode, so comparing our TokensPerSecond against their
// decode_tok_s would be unfair to US — ours carries the prefill and theirs does not. Here the stopwatches are
// placed by hand so both engines are read the same way.
//
// Output is one JSON line with dotLLM's field names, so one parser reads both.

using System;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Contracts;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("usage: GenHarness <model.gguf> <prompt> <maxTokens> [threads]");

            return 2;
        }

        var modelPath = args[0];
        var prompt = args[1];
        var maxTokens = int.Parse(args[2], CultureInfo.InvariantCulture);

        // Applied BEFORE anything reads ProcessorCount, exactly as ProfHarness does: a thread-count knob set
        // after the pool has sized itself is a knob that does nothing, and the arm then runs identical to the
        // one it was supposed to differ from.
        if (args.Length > 3 && args[3] != "default")
        {
            // The NAMES were verified against OverfitEnvironment rather than guessed. The first version of
            // this harness invented OVERFIT_DECODE_MAX_WORKERS and OVERFIT_MAX_WORKERS, neither of which
            // exists — the arm would have run identical to the default one and the flat result would have
            // been read as "worker count is not the cause".
            Environment.SetEnvironmentVariable("OVERFIT_DECODE_WORKERS", args[3]);
            Environment.SetEnvironmentVariable("OVERFIT_PARALLEL_WORKERS", args[3]);
        }

        // Memory checkpoints. The question is not "how much" — that is measured from outside — but WHEN it is
        // committed and whether it lands on the managed heap. Private commit minus managed heap is the
        // unmanaged part: pooled buffers, NativeMemory, and anything the mapping forces resident.
        Mark("start");

        // REMOVED 2026-08-21: a `GgufTokenizer.Load` probe stood here, added the day before to test whether
        // the tokenizer explained an unattributed gigabyte. It did not — the hypothesis was refuted the same
        // hour — but the probe was left behind, and it holds ~40 MB for the life of the process.
        //
        // **Every load-time figure this harness reported for a day was 40 MB high**, including a 727.2 MB
        // reading that was quoted in `measured-baselines.md` and a 759.2 that preceded it. Nothing about
        // those numbers looked wrong; the error surfaced only when the measurement was split into parts that
        // had to add up, and the parts came to 40 MB more than the whole.
        //
        // The lesson is not "remove probes when done" — it is that a probe which allocates is part of the
        // subject from the moment it exists. If one is needed again, take the reading with and without it in
        // the same sitting, or measure the tokenizer from inside `LoadGguf` where it is already being built.
        var loadSw = Stopwatch.StartNew();
        using var client = OverfitClient.LoadGguf(modelPath, mmap: true);
        loadSw.Stop();

        Mark("after LoadGguf");

        // The Contracts tokenizer is span-based, so the buffer is sized by CountTokens first. A fixed
        // guess here would silently truncate a longer prompt and the comparison would then be between two
        // different inputs.
        var promptBuffer = new int[client.Tokenizer.CountTokens(prompt)];
        var promptLength = client.Tokenizer.Encode(prompt, promptBuffer);
        var promptTokens = promptBuffer.AsSpan(0, promptLength);

        // Sized exactly as OverfitClient sizes its own, so this harness stops measuring a configuration the
        // product does not ship. The earlier version called CreateSession() with no argument, which resolves
        // to the MODEL's full context.
        using var session = client.Engine.CreateSession(2048);

        Mark("after CreateSession(2048)");

        var sampling = SamplingOptions.Greedy;

        var prefillSw = Stopwatch.StartNew();
        session.Prefill(promptTokens);
        prefillSw.Stop();

        var generated = new int[maxTokens];
        var count = 0;

        var decodeSw = Stopwatch.StartNew();

        for (var i = 0; i < maxTokens; i++)
        {
            generated[count++] = session.GenerateNextToken(in sampling);
        }

        decodeSw.Stop();

        Mark("after decode");

        var text = client.Tokenizer.DecodeToString(generated.AsSpan(0, count));

        var loadMs = loadSw.Elapsed.TotalMilliseconds;
        var prefillMs = prefillSw.Elapsed.TotalMilliseconds;
        var decodeMs = decodeSw.Elapsed.TotalMilliseconds;

        // Sampling is inside GenerateNextToken and is not separable without instrumenting the runtime, so it
        // is reported as 0 and folded into decode. That makes our decode number the PESSIMISTIC one against
        // dotLLM, which reports sampling separately — worth stating rather than quietly benefiting from.
        var json = new StringBuilder();

        json.Append('{');
        json.Append("\"engine\":\"overfit\",");
        json.Append("\"text\":").Append(Quote(text)).Append(',');
        json.Append("\"prompt\":").Append(Quote(prompt)).Append(',');
        json.Append("\"usage\":{\"prompt_tokens\":").Append(promptLength)
            .Append(",\"generated_tokens\":").Append(count).Append("},");
        json.Append("\"timings\":{");
        json.Append("\"load_ms\":").Append(F(loadMs)).Append(',');
        json.Append("\"prefill_ms\":").Append(F(prefillMs)).Append(',');
        json.Append("\"decode_ms\":").Append(F(decodeMs)).Append(',');
        json.Append("\"sampling_ms\":0,");
        json.Append("\"prefill_tok_s\":")
            .Append(F(prefillMs > 0 ? promptLength / (prefillMs / 1000.0) : 0)).Append(',');
        json.Append("\"decode_tok_s\":").Append(F(decodeMs > 0 ? count / (decodeMs / 1000.0) : 0));
        json.Append("},");
        // BOTH pools are echoed. Reporting only the decode pool is what hid the real difference between two
        // readings six-fold apart: passing an explicit thread count sets the GENERAL pool too, while the
        // default leaves it at ProcessorCount (32 here) against 16 physical cores. A run whose lever is not
        // fully visible cannot be compared with another run.
        json.Append("\"decode_workers\":").Append(DevOnBike.Overfit.Runtime.OverfitParallel.DecodeMaxWorkers);
        json.Append(",\"general_workers\":").Append(DevOnBike.Overfit.Runtime.OverfitParallel.WorkerCount);
        json.Append(",\"requested_workers\":").Append(Quote(args.Length > 3 ? args[3] : "default"));
        json.Append('}');

        Console.WriteLine(json.ToString());

        return 0;
    }

    private static void Mark(string label)
    {
        using var self = System.Diagnostics.Process.GetCurrentProcess();

        self.Refresh();

        var privateMb = self.PrivateMemorySize64 / (1024.0 * 1024.0);
        var workingMb = self.WorkingSet64 / (1024.0 * 1024.0);
        var managedMb = GC.GetTotalMemory(false) / (1024.0 * 1024.0);

        Console.Error.WriteLine(
            $"MARK {label,-28} private {privateMb,9:F1} MB   working {workingMb,9:F1} MB   "
            + $"managed {managedMb,9:F1} MB   unmanaged {privateMb - managedMb,9:F1} MB");
    }

    private static string F(double value) => value.ToString("F3", CultureInfo.InvariantCulture);

    private static string Quote(string value)
    {
        var sb = new StringBuilder("\"");

        foreach (var c in value)
        {
            if (c == '"' || c == '\\')
            {
                sb.Append('\\').Append(c);

                continue;
            }

            if (c < ' ')
            {
                sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));

                continue;
            }

            sb.Append(c);
        }

        return sb.Append('"').ToString();
    }
}
