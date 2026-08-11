
// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;

namespace DevOnBike.Overfit.Audio
{
    /// <summary>
    /// Minimal, dependency-free WAV reader for the speech front-end: parses a RIFF/WAVE file and returns
    /// mono float samples in [−1, 1]. Supports 16-bit PCM and 32-bit IEEE float; multi-channel audio is
    /// down-mixed to mono by averaging. No resampling — the caller should feed 16 kHz audio (Whisper's rate).
    /// </summary>
    public static class WavReader
    {
        /// <summary>Reads <paramref name="path"/> → mono float samples; <paramref name="sampleRate"/> receives the rate (Hz).</summary>
        public static float[] ReadMono(string path, out int sampleRate)
        {
            using var fs = File.OpenRead(path);
            return ReadMono(fs, out sampleRate);
        }

        /// <summary>Reads a WAV stream → mono float samples.</summary>
        public static float[] ReadMono(Stream stream, out int sampleRate)
        {
            using var br = new BinaryReader(stream);
            if (ReadTag(br) != "RIFF")
            {
                throw new OverfitFormatException("Not a RIFF file.");
            }
            br.ReadInt32(); // file size
            if (ReadTag(br) != "WAVE")
            {
                throw new OverfitFormatException("Not a WAVE file.");
            }

            int channels = 0, bitsPerSample = 0, audioFormat = 0;
            sampleRate = 0;
            byte[]? data = null;

            while (stream.Position < stream.Length)
            {
                var chunkId = ReadTag(br);
                var chunkSize = ReadChunkSize(br, stream, chunkId);
                if (chunkId == "fmt ")
                {
                    audioFormat = br.ReadInt16();   // 1 = PCM, 3 = IEEE float
                    channels = br.ReadInt16();
                    sampleRate = br.ReadInt32();
                    br.ReadInt32();                 // byte rate
                    br.ReadInt16();                 // block align
                    bitsPerSample = br.ReadInt16();
                    var consumed = 16;
                    if (chunkSize > consumed)
                    {
                        br.ReadBytes(chunkSize - consumed);
                    } // skip extension
                }
                if (chunkId == "data")
                {
                    data = br.ReadBytes(chunkSize);

                    // ReadBytes returns what it got, and a header that claims more than the file holds used
                    // to leave shorter audio with nothing said about it. That is the worse half of this
                    // defect: shorter audio is not obviously wrong, and everything downstream - transcription,
                    // similarity scoring - accepts it and produces a plausible answer about a truncated file.
                    if (data.Length != chunkSize)
                    {
                        throw new OverfitFormatException(
                            $"The WAV data chunk claims {chunkSize} bytes and the file holds {data.Length}. "
                            + "Reading on would silently transcribe truncated audio.");
                    }
                }

                if (chunkId != "fmt " && chunkId != "data")
                {
                    br.ReadBytes(chunkSize);        // skip unknown chunk
                    if ((chunkSize & 1) == 1)
                    {
                        br.ReadByte();
                    } // chunks are word-aligned
                }
            }

            if (data == null || channels == 0)
            {
                throw new OverfitFormatException("WAV missing fmt/data chunk.");
            }

            return Decode(data, audioFormat, channels, bitsPerSample);
        }

        // OVERFIT001: by-contract — decodes the WAV payload into a fresh PCM array the caller owns (one
        // allocation per file load, not a per-frame hot path); the down-mix buffer is likewise the output.
#pragma warning disable OVERFIT001
        private static float[] Decode(byte[] data, int audioFormat, int channels, int bitsPerSample)
        {
            var span = data.AsSpan();

            var is16BitPcm = audioFormat == 1 && bitsPerSample == 16;
            var is32BitFloat = audioFormat == 3 && bitsPerSample == 32;

            // Reject up front, so the two supported paths below leave `interleaved` definitely assigned.
            if (!is16BitPcm && !is32BitFloat)
            {
                throw new OverfitRuntimeException($"Unsupported WAV format (audioFormat={audioFormat}, bits={bitsPerSample}). Use 16-bit PCM or 32-bit float.");
            }

            var bytesPerSample = is16BitPcm ? 2 : 4;
            var interleaved = new float[data.Length / bytesPerSample];

            if (is16BitPcm)
            {
                for (var i = 0; i < interleaved.Length; i++)
                {
                    interleaved[i] = BinaryPrimitives.ReadInt16LittleEndian(span.Slice(i * 2, 2)) / 32768f;
                }
            }

            if (is32BitFloat)
            {
                for (var i = 0; i < interleaved.Length; i++)
                {
                    interleaved[i] = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(i * 4, 4));
                }
            }

            if (channels == 1)
            {
                return interleaved;
            }

            // Down-mix to mono by averaging channels.
            var frames = interleaved.Length / channels;
            var mono = new float[frames];
            for (var i = 0; i < frames; i++)
            {
                var acc = 0f;
                for (var c = 0; c < channels; c++)
                {
                    acc += interleaved[i * channels + c];
                }
                mono[i] = acc / channels;
            }
            return mono;
        }
#pragma warning restore OVERFIT001

        /// <summary>
        /// A chunk length, checked against the file rather than trusted.
        ///
        /// <para>The size is a 32-bit field inside an envelope this reader has already accepted, and it was
        /// used directly in buffer arithmetic. A negative value reached <c>ReadBytes</c> and surfaced as a raw
        /// <c>ArgumentOutOfRangeException</c> - not this project's format exception, so a caller catching
        /// malformed input did not catch it - and an oversized-but-positive one truncated the audio in
        /// silence. Same root cause as the big_values field in <c>Mp3Decoder</c>: a length taken from inside a
        /// validated envelope and used without a second check.</para>
        /// </summary>
        private static int ReadChunkSize(BinaryReader br, Stream stream, string chunkId)
        {
            var size = br.ReadInt32();

            if (size < 0)
            {
                throw new OverfitFormatException(
                    $"WAV chunk '{chunkId}' declares a negative size ({size}). The file is malformed.");
            }

            var remaining = stream.Length - stream.Position;

            if (size > remaining)
            {
                throw new OverfitFormatException(
                    $"WAV chunk '{chunkId}' declares {size} bytes with only {remaining} left in the file.");
            }

            return size;
        }

        private static string ReadTag(BinaryReader br)
        {
            Span<byte> tag = stackalloc byte[4];
            if (br.Read(tag) != 4)
            {
                throw new EndOfStreamException("Unexpected end of WAV.");
            }
            return System.Text.Encoding.ASCII.GetString(tag);
        }
    }
}
