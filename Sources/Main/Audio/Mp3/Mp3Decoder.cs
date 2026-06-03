// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Mp3
{
    /// <summary>
    /// MPEG-1/2/2.5 Layer III audio decoder (work in progress). The container/bitstream layer
    /// (<see cref="Mp3FrameHeader"/>, <see cref="Mp3BitReader"/>, frame walking) is complete and validated;
    /// the per-frame Layer III synthesis (bit reservoir → scalefactors → Huffman → requantize → reorder →
    /// stereo → antialias → IMDCT → subband synthesis) is being filled in stage by stage.
    /// </summary>
    internal sealed class Mp3Decoder
    {
        /// <summary>Decodes the whole stream to mono 32-bit float PCM in [-1, 1].</summary>
        public float[] DecodeMono(byte[] bytes, out int sampleRate)
        {
            var info = Mp3Reader.Probe(bytes);
            sampleRate = info.SampleRate;
            throw new NotSupportedException(
                "MP3 Layer III audio synthesis is not yet implemented (container layer is in place). " +
                $"Probed: {info.SampleRate} Hz, {info.Channels} ch, {info.FrameCount} frames, {info.DurationSeconds:F2}s.");
        }
    }
}
