// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.Audio;

namespace DevOnBike.Overfit.Demo.VoiceLoop
{
    /// <summary>
    /// Plays mono float PCM through the default output device using the built-in Windows <c>winmm</c>
    /// <c>PlaySound</c> API (P/Invoke — no NuGet). Writes a temp WAV via <see cref="WavWriter"/> and plays it
    /// synchronously. Best-effort: on a headless box with no audio device it simply no-ops. Demo-only.
    /// </summary>
    internal static class AudioPlayer
    {
        private const uint SndSync = 0x0000;
        private const uint SndFilename = 0x00020000;

        [DllImport("winmm.dll", CharSet = CharSet.Unicode)]
        private static extern bool PlaySound(string? pszSound, IntPtr hmod, uint fdwSound);

        public static void Play(float[] samples, int sampleRate)
        {
            var path = Path.Combine(Path.GetTempPath(), "overfit_voiceloop_play.wav");
            WavWriter.WriteMono(path, samples, sampleRate, WavSampleFormat.Pcm16);
            
            try
            {
                PlaySound(path, IntPtr.Zero, SndFilename | SndSync);
            }
            catch
            {
                // No audio device (headless) — playback is best-effort.
            }
        }
    }
}
