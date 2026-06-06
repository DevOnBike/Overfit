// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Demo.VoiceLoop
{
    /// <summary>
    /// Microphone capture via the built-in Windows <c>winmm</c> <c>waveIn</c> API (P/Invoke — no NuGet, no native
    /// binary shipped). Records a fixed number of seconds of 16 kHz mono 16-bit PCM and returns it as float in
    /// [-1, 1], ready for the Whisper frontend. Demo-only; the core engine stays platform-neutral.
    /// </summary>
    internal static class MicCapture
    {
        public const int SampleRate = 16000;
        private const uint WaveMapper = 0xFFFFFFFF;
        private const ushort WaveFormatPcm = 1;

        [StructLayout(LayoutKind.Sequential)]
        private struct WaveFormatEx
        {
            public ushort wFormatTag;
            public ushort nChannels;
            public uint nSamplesPerSec;
            public uint nAvgBytesPerSec;
            public ushort nBlockAlign;
            public ushort wBitsPerSample;
            public ushort cbSize;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct WaveHdr
        {
            public IntPtr lpData;
            public uint dwBufferLength;
            public uint dwBytesRecorded;
            public IntPtr dwUser;
            public uint dwFlags;
            public uint dwLoops;
            public IntPtr lpNext;
            public IntPtr reserved;
        }

        [DllImport("winmm.dll")]
        private static extern int waveInOpen(out IntPtr phwi, uint uDeviceID, ref WaveFormatEx pwfx, IntPtr cb, IntPtr inst, uint flags);
        [DllImport("winmm.dll")]
        private static extern int waveInClose(IntPtr hwi);
        [DllImport("winmm.dll")]
        private static extern int waveInPrepareHeader(IntPtr hwi, IntPtr pwh, uint cb);
        [DllImport("winmm.dll")]
        private static extern int waveInUnprepareHeader(IntPtr hwi, IntPtr pwh, uint cb);
        [DllImport("winmm.dll")]
        private static extern int waveInAddBuffer(IntPtr hwi, IntPtr pwh, uint cb);
        [DllImport("winmm.dll")]
        private static extern int waveInStart(IntPtr hwi);
        [DllImport("winmm.dll")]
        private static extern int waveInStop(IntPtr hwi);
        [DllImport("winmm.dll")]
        private static extern int waveInReset(IntPtr hwi);

        /// <summary>Records <paramref name="seconds"/> of audio from the default input device → mono 16 kHz float.</summary>
        public static float[] Record(int seconds)
        {
            var fmt = new WaveFormatEx
            {
                wFormatTag = WaveFormatPcm,
                nChannels = 1,
                nSamplesPerSec = SampleRate,
                wBitsPerSample = 16,
                nBlockAlign = 2,
                nAvgBytesPerSec = SampleRate * 2,
                cbSize = 0,
            };

            if (waveInOpen(out var h, WaveMapper, ref fmt, IntPtr.Zero, IntPtr.Zero, 0) != 0)
            {
                throw new InvalidOperationException("Could not open the default microphone (waveInOpen failed).");
            }

            var byteCount = seconds * SampleRate * 2;
            var buffer = new byte[byteCount];
            using var gch = GCHandleScope.Pin(buffer);
            var hdrSize = (uint)Marshal.SizeOf<WaveHdr>();
            var pHdr = Marshal.AllocHGlobal((int)hdrSize);
            try
            {
                var hdr = new WaveHdr { lpData = gch.Address, dwBufferLength = (uint)byteCount };
                Marshal.StructureToPtr(hdr, pHdr, false);

                waveInPrepareHeader(h, pHdr, hdrSize);
                waveInAddBuffer(h, pHdr, hdrSize);
                waveInStart(h);
                Thread.Sleep((seconds * 1000) + 200);
                waveInStop(h);
                waveInReset(h);
                waveInUnprepareHeader(h, pHdr, hdrSize);

                hdr = Marshal.PtrToStructure<WaveHdr>(pHdr);
                var recorded = (int)hdr.dwBytesRecorded;

                var samples = new float[recorded / 2];
                for (var i = 0; i < samples.Length; i++)
                {
                    var s = (short)(buffer[i * 2] | (buffer[(i * 2) + 1] << 8));
                    samples[i] = s / 32768f;
                }
                return samples;
            }
            finally
            {
                waveInClose(h);
                Marshal.FreeHGlobal(pHdr);
            }
        }
    }
}
