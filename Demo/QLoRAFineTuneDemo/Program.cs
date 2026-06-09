// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.LanguageModels.LoRA;

namespace DevOnBike.Overfit.Demo.QLoRAFineTune
{
    /// <summary>
    /// QLoRA fine-tune demo — teach a quantized LLM new knowledge on your CPU, in .NET.
    ///
    ///   QLoRAFineTuneDemo &lt;model.gguf&gt; &lt;text.txt&gt; [epochs]      → fine-tune then chat
    ///   QLoRAFineTuneDemo &lt;model.gguf&gt; --adapter &lt;adapter.lora&gt; → chat with a saved adapter
    ///
    /// The .gguf's directory must also contain tokenizer.json. No GPU, no Python.
    /// </summary>
    internal static class Program
    {
        private static int Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Usage:");
                Console.WriteLine("  QLoRAFineTuneDemo <model.gguf> <text.txt> [epochs]      fine-tune on the text, then chat");
                Console.WriteLine("  QLoRAFineTuneDemo <model.gguf> --adapter <adapter.lora> load a saved adapter, then chat");
                return 1;
            }

            var ggufPath = args[0];
            if (!File.Exists(ggufPath))
            {
                Console.Error.WriteLine($"GGUF not found: {ggufPath}");
                return 1;
            }

            var chatOnly = args[1] == "--adapter";
            var epochs = args.Length >= 3 && int.TryParse(args[2], out var e) ? e : 3;

            Console.WriteLine($"Loading {Path.GetFileName(ggufPath)} (4-bit base + tokenizer) ...");
            var sw = Stopwatch.StartNew();
            using var tuner = new QLoRAFineTuner(ggufPath, new QLoRAOptions { Epochs = epochs });
            Console.WriteLine($"  loaded {tuner.LayerCount} layers in {sw.Elapsed.TotalSeconds:F1}s\n");

            if (chatOnly)
            {
                var adapterPath = args[2];
                Console.WriteLine($"Loading adapter {adapterPath} ...");
                tuner.LoadAdapter(adapterPath);
                Console.WriteLine("  ready.\n");
            }
            else
            {
                var textPath = args[1];
                if (!File.Exists(textPath))
                {
                    Console.Error.WriteLine($"Text file not found: {textPath}");
                    return 1;
                }

                var text = File.ReadAllText(textPath);
                Console.WriteLine($"Fine-tuning on {Path.GetFileName(textPath)} ({text.Length:N0} chars), {epochs} epoch(s) ...");

                sw.Restart();
                var lastLog = 0L;
                var history = tuner.FineTune(text, onStep: (epoch, step, loss) =>
                {
                    // Throttle the console to ~2 lines/sec.
                    if (sw.ElapsedMilliseconds - lastLog >= 500 || step == 0)
                    {
                        lastLog = sw.ElapsedMilliseconds;
                        Console.Write($"\r  epoch {epoch + 1}/{epochs}  step {step,4}  loss {loss,8:F4}   ");
                    }
                });
                Console.WriteLine($"\n  done: loss {history[0]:F3} -> {history[^1]:F4} over {history.Count} steps ({sw.Elapsed.TotalMinutes:F1} min)");

                var savePath = Path.ChangeExtension(textPath, ".lora");
                tuner.SaveAdapter(savePath);
                Console.WriteLine($"  adapter saved: {savePath} ({new FileInfo(savePath).Length / (1024.0 * 1024):F1} MB; the base GGUF is untouched)\n");
            }

            // ── interactive chat ──
            Console.WriteLine("Ask the model anything (it now knows your text). Empty line or 'exit' to quit.\n");

            while (true)
            {
                Console.Write("you> ");
                var prompt = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(prompt) || prompt.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase))
                {
                    break;
                }

                var gen = Stopwatch.StartNew();
                var answer = tuner.Ask(prompt, maxNewTokens: 48);

                gen.Stop();

                Console.WriteLine($"bot> {answer.Trim()}");
                Console.WriteLine($"     ({gen.Elapsed.TotalSeconds:F1}s)\n");
            }

            Console.WriteLine("bye.");
            return 0;
        }
    }
}
