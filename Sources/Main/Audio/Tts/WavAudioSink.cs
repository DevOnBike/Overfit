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
            _infoComment = metadata?.ToInfoComment();
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
