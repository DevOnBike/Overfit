// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Exceptions;
using DevOnBike.Overfit.LanguageModels.Whisper;

namespace DevOnBike.Overfit.Demo.WhisperConsole
{
    /// <summary>
    /// Whisper speech-to-text demo — transcribe a WAV or MP3 with a whisper.cpp ggml model, in pure .NET on the CPU.
    ///
    ///   WhisperDemo &lt;model.ggml.bin&gt; &lt;audio.wav|audio.mp3&gt; [language]
    ///
    /// e.g.  WhisperDemo C:\whisper\ggml-tiny.bin C:\whisper\jfk.wav en
    ///       WhisperDemo C:\whisper\ggml-tiny.bin C:\whisper\pl.mp3 pl
    ///
    /// WAV (16-bit PCM / 32-bit float) and MP3 (MPEG-1/2/2.5 Layer III) are decoded in pure C#; any sample rate
    /// is resampled to 16 kHz, stereo is downmixed to mono. No GPU, no Python, no external libraries.
    /// </summary>
    internal static class Program
    {
        private static int Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: WhisperDemo <model.ggml.bin> <audio.wav|audio.mp3> [language=en]");
                Console.WriteLine("  language: en, pl, de, es, ... (Whisper language code; ignored for English-only models)");
                return 1;
            }

            var modelPath = args[0];
            var wavPath = args[1];
            var language = args.Length >= 3 ? args[2] : "en";

            if (!File.Exists(modelPath))
            {
                Console.Error.WriteLine($"Model not found: {modelPath}");
                return 1;
            }
            if (!File.Exists(wavPath))
            {
                Console.Error.WriteLine($"Audio not found: {wavPath}");
                return 1;
            }

            Console.WriteLine($"Loading {Path.GetFileName(modelPath)} ...");
            var sw = Stopwatch.StartNew();
            var whisper = WhisperTranscriber.Load(modelPath);
            var c = whisper.Config;
            Console.WriteLine($"  {c.NAudioLayer}+{c.NTextLayer} layers, state {c.NAudioState}, {(c.IsMultilingual ? "multilingual" : "English-only")} — loaded in {sw.Elapsed.TotalSeconds:F1}s\n");

            Console.WriteLine($"Transcribing {Path.GetFileName(wavPath)} (language: {language}) ...");
            sw.Restart();
            string text;
            try
            {
                text = whisper.TranscribeFile(wavPath, language);
            }
            catch (OverfitRuntimeException ex)
            {
                Console.Error.WriteLine($"  {ex.Message}");
                Console.Error.WriteLine("  Tip: convert to 16 kHz mono WAV, e.g.  ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav");
                return 1;
            }
            sw.Stop();

            Console.WriteLine($"  done in {sw.Elapsed.TotalSeconds:F1}s\n");
            Console.WriteLine("─────────────────────────────────────────────");
            Console.WriteLine(text);
            Console.WriteLine("─────────────────────────────────────────────");
            return 0;
        }
    }
}
