// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Mp3;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>
    /// Validation of the MP3 container/bitstream layer (frame walking + header parsing) on a real file —
    /// version / sample rate / channels / frame count / duration. This is the first checkpoint before the
    /// Layer III audio synthesis lands. <see cref="LongFactAttribute"/> — needs a real <c>jfk.mp3</c>.
    /// </summary>
    public sealed class Mp3ReaderTests
    {
        private readonly ITestOutputHelper _out;
        public Mp3ReaderTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Decode_RealMp3_Stats()
        {
            var samples = Mp3Reader.ReadMono(TestModelPaths.Whisper.RequireSampleMp3Path(), out var sr);
            double sum = 0, sumSq = 0, min = double.MaxValue, max = double.MinValue;
            var nan = 0;
            foreach (var s in samples)
            {
                if (!float.IsFinite(s)) { nan++; continue; }
                sum += s; sumSq += (double)s * s;
                if (s < min) { min = s; }
                if (s > max) { max = s; }
            }
            var rms = Math.Sqrt(sumSq / samples.Length);
            _out.WriteLine($"sampleRate {sr}, samples {samples.Length}, rms {rms:F4}, min {min:F4}, max {max:F4}, nan {nan}");
            Assert.Equal(0, nan);
            Assert.True(rms > 0.001, $"output looks like silence (rms {rms})");
            Assert.True(max <= 1.5 && min >= -1.5, "output grossly out of range");
        }

        [LongFact]
        public void Decode_PerFrame_ZeroAlloc()
        {
            var bytes = File.ReadAllBytes(TestModelPaths.Whisper.RequireSampleMp3Path());
            var dec = new Mp3Decoder();
            dec.DecodeMono(bytes, out _); // warm up: JIT + first allocation of instance scratch

            var before = GC.GetAllocatedBytesForCurrentThread();
            var samples = dec.DecodeMono(bytes, out _);
            var after = GC.GetAllocatedBytesForCurrentThread();

            var alloc = after - before;
            var outputBytes = (long)samples.Length * sizeof(float);
            var overhead = alloc - outputBytes;
            _out.WriteLine($"total alloc {alloc} B, output buffer {outputBytes} B, per-decode overhead {overhead} B");
            // The only heap allocation in a decode is the single output buffer; all per-frame working buffers
            // are pre-allocated instance fields, so the overhead beyond the output must be ~zero.
            Assert.True(overhead < 4096, $"decode allocates beyond the output buffer ({overhead} B over {samples.Length} samples)");
        }

        [LongFact]
        public void Probe_RealMp3_ReportsConsistentContainerMetadata()
        {
            var bytes = File.ReadAllBytes(TestModelPaths.Whisper.RequireSampleMp3Path());
            var info = Mp3Reader.Probe(bytes);

            _out.WriteLine($"sampleRate : {info.SampleRate} Hz");
            _out.WriteLine($"channels   : {info.Channels}");
            _out.WriteLine($"frames     : {info.FrameCount}");
            _out.WriteLine($"samples/ch : {info.SampleCount}");
            _out.WriteLine($"duration   : {info.DurationSeconds:F3} s");

            Assert.True(info.FrameCount > 0, "should walk at least one valid frame");
            Assert.True(info.SampleRate is 8000 or 11025 or 12000 or 16000 or 22050 or 24000 or 32000 or 44100 or 48000,
                $"unexpected sample rate {info.SampleRate}");
            Assert.InRange(info.Channels, 1, 2);
            // The JFK clip is ~11 s; sanity-bound the probed duration loosely.
            Assert.InRange(info.DurationSeconds, 1.0, 60.0);
        }
    }
}
