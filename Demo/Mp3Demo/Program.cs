// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Audio.Mp3;

namespace DevOnBike.Overfit.Demo.Mp3Console
{
    /// <summary>
    /// Pure-C# MP3 decoder demo — decodes an MPEG-1/2/2.5 Layer III file to mono 32-bit float PCM and (by
    /// default) writes a 16 kHz 16-bit WAV next to it, so you can hear that the from-scratch decoder works.
    /// No native binaries, no external libraries, no Python.
    ///
    ///   Mp3Demo &lt;input.mp3&gt; [output.wav]
    ///
    /// e.g.  Mp3Demo C:\whisper\pl.mp3
    ///       Mp3Demo C:\music\song.mp3 C:\music\song.wav
    /// </summary>
    internal static class Program
    {
        private const int TargetRate = 16000;

        private static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: Mp3Demo <input.mp3> [output.wav]");
                Console.WriteLine("  Decodes the MP3 to mono PCM and writes a 16 kHz 16-bit WAV (default: <input>.wav).");
                return 1;
            }

            var inPath = args[0];
            if (!File.Exists(inPath))
            {
                Console.Error.WriteLine($"File not found: {inPath}");
                return 1;
            }
            var outPath = args.Length >= 2 ? args[1] : Path.ChangeExtension(inPath, ".wav");

            Console.WriteLine($"Decoding {Path.GetFileName(inPath)} ...");
            var sw = Stopwatch.StartNew();
            var samples = Mp3Reader.ReadMono(inPath, out var sampleRate);
            sw.Stop();

            var rms = Rms(samples);
            var duration = sampleRate > 0 ? (double)samples.Length / sampleRate : 0;
            Console.WriteLine($"  {sampleRate} Hz mono, {samples.Length:N0} samples, {duration:F2} s, RMS {rms:F4}");
            Console.WriteLine($"  decoded in {sw.Elapsed.TotalMilliseconds:F0} ms (pure C#, zero per-frame allocations)\n");

            // Resample to 16 kHz so the output is a tidy, widely-playable WAV.
            var resampled = sampleRate == TargetRate
                ? samples
                : Audio.AudioResampler.Resample(samples, sampleRate, TargetRate);
            WriteWav16(outPath, resampled, TargetRate);
            Console.WriteLine($"Wrote {outPath} ({TargetRate} Hz mono 16-bit, {resampled.Length:N0} samples).");
            return 0;
        }

        private static double Rms(ReadOnlySpan<float> x)
        {
            if (x.Length == 0)
            {
                return 0;
            }
            double sum = 0;
            foreach (var v in x)
            {
                sum += (double)v * v;
            }
            return Math.Sqrt(sum / x.Length);
        }

        private static void WriteWav16(string path, ReadOnlySpan<float> samples, int sampleRate)
        {
            using var fs = File.Create(path);
            using var w = new BinaryWriter(fs);
            var dataBytes = samples.Length * 2;
            // RIFF / WAVE header (PCM, mono, 16-bit)
            w.Write("RIFF"u8);
            w.Write(36 + dataBytes);
            w.Write("WAVE"u8);
            w.Write("fmt "u8);
            w.Write(16);              // fmt chunk size
            w.Write((short)1);        // PCM
            w.Write((short)1);        // channels
            w.Write(sampleRate);
            w.Write(sampleRate * 2);  // byte rate
            w.Write((short)2);        // block align
            w.Write((short)16);       // bits per sample
            w.Write("data"u8);
            w.Write(dataBytes);
            foreach (var s in samples)
            {
                var clamped = s < -1f ? -1f : s > 1f ? 1f : s;
                w.Write((short)(clamped * 32767f));
            }
        }
    }
}
