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
