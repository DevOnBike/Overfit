// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Whisper;

namespace DevOnBike.Overfit.Demo.MicConsole
{
    /// <summary>
    /// Live microphone speech-to-text — record a few seconds from the mic and transcribe with Whisper, pure
    /// .NET on the CPU. Mic capture uses the built-in Windows winmm API (no NuGet). Record-then-transcribe loop;
    /// the transcriber reuses its buffers, so each round is allocation-stable.
    ///
    ///   MicDemo &lt;model.ggml.bin&gt; [language=en] [seconds=5]
    ///
    /// e.g.  MicDemo C:\whisper\ggml-tiny.bin pl 5
    /// </summary>
    internal static class Program
    {
        private static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: MicDemo <model.ggml.bin> [language=en] [seconds=5]");
                return 1;
            }
            if (!OperatingSystem.IsWindows())
            {
                Console.Error.WriteLine("MicDemo uses the Windows winmm API; run it on Windows (or feed a file to WhisperDemo/Mp3Demo).");
                return 1;
            }

            var modelPath = args[0];
            var language = args.Length >= 2 ? args[1] : "en";
            var seconds = args.Length >= 3 && int.TryParse(args[2], out var s) ? s : 5;
            if (!File.Exists(modelPath)) { Console.Error.WriteLine($"Model not found: {modelPath}"); return 1; }

            Console.WriteLine($"Loading {Path.GetFileName(modelPath)} ...");
            var whisper = WhisperTranscriber.Load(modelPath);
            Console.WriteLine($"Ready. Language: {language}. Records {seconds}s per round.\n");

            while (true)
            {
                Console.Write("Press Enter to record (or 'q' + Enter to quit): ");
                var line = Console.ReadLine();
                if (line is not null && line.Trim().Equals("q", StringComparison.OrdinalIgnoreCase))
                {
                    break;
                }

                Console.WriteLine($"  recording {seconds}s ...");
                var samples = MicCapture.Record(seconds);

                var sw = Stopwatch.StartNew();
                var text = whisper.Transcribe(samples, language);
                sw.Stop();

                Console.WriteLine($"  ({sw.Elapsed.TotalSeconds:F1}s)  ▶  {text}\n");
            }
            return 0;
        }
    }
}
