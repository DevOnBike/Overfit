// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>
    /// An <see cref="IAudioSink"/> that collects the streamed PCM and writes one mono WAV on
    /// <see cref="Complete"/> (via <see cref="WavWriter"/>), embedding the synthetic-speech provenance marker if
    /// supplied. Buffer-then-write keeps the header correct on a non-seekable stream; for true real-time streaming
    /// a header-patching sink comes later. Disposing completes if not already completed.
    /// </summary>
    public sealed class WavAudioSink : IAudioSink, IDisposable
    {
        private readonly Stream _output;
        private readonly bool _leaveOpen;
        private readonly WavSampleFormat _format;
        private readonly string? _infoComment;
        private readonly List<float> _buffer = [];
        private bool _completed;
        private bool _disposed;

        /// <param name="metadata">
        /// The synthetic-speech marker written into the file's INFO chunk. <b>Omitting it now marks the file
        /// anyway</b>, with a marker naming no voice profile.
        ///
        /// <para>It used to mean "write nothing", and that was the wrong direction for a default to fail in.
        /// This sink lives in the TTS namespace: everything it writes is generated speech, and a
        /// voice-cloning path able to produce unmarked audio of a real person is the one property in this
        /// repository that must not be merely intended. It was intended - <c>Sources/Main/Audio/README.md</c>
        /// asserted the marker was enforced at the engine, and the engine never referenced it; the three call
        /// sites that passed it were the whole of the enforcement, so any new caller got unmarked output by
        /// default.</para>
        ///
        /// <para>To write a genuinely unmarked file - a decision, not an omission - pass
        /// <see cref="SyntheticSpeechMetadata.Unmarked"/> and say why at the call site.</para>
        /// </param>
        /// <param name="output">Destination stream the WAV bytes are written to.</param>
        /// <param name="sampleRate">Sample rate in Hz of the PCM this sink will be given. Must be positive.</param>
        /// <param name="format">Sample encoding written into the header and used when converting incoming samples.</param>
        /// <param name="leaveOpen">When <c>true</c> the stream survives this sink's disposal; the default closes it.</param>
        public WavAudioSink(
            Stream output,
            int sampleRate,
            WavSampleFormat format = WavSampleFormat.Pcm16,
            SyntheticSpeechMetadata? metadata = null,
            bool leaveOpen = false)
        {
            ArgumentNullException.ThrowIfNull(output);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
            _output = output;
            SampleRate = sampleRate;
            _format = format;
            _infoComment = (metadata ?? SyntheticSpeechMetadata.ForNow(null)).ToInfoComment();
            _leaveOpen = leaveOpen;
        }

        public WavAudioSink(
            string path,
            int sampleRate,
            WavSampleFormat format = WavSampleFormat.Pcm16,
            SyntheticSpeechMetadata? metadata = null)
            : this(File.Create(path), sampleRate, format, metadata, leaveOpen: false)
        {
        }

        public int SampleRate
        {
            get;
        }

        public void Write(ReadOnlySpan<float> samples)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_completed)
            {
                throw new OverfitRuntimeException("Cannot write to a completed audio sink.");
            }
            for (var i = 0; i < samples.Length; i++)
            {
                _buffer.Add(samples[i]);
            }
        }

        public void Complete()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_completed)
            {
                return;
            }
            WavWriter.WriteMono(_output, CollectionsMarshal.AsSpan(_buffer), SampleRate, _format, _infoComment);
            _output.Flush();
            _completed = true;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            if (!_completed)
            {
                Complete();
            }
            _disposed = true;
            if (!_leaveOpen)
            {
                _output.Dispose();
            }
        }
    }
}
