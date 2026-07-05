// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Android.Media;

namespace DevOnBike.OverfitChat
{
    /// <summary>
    /// Captures microphone audio as 16 kHz mono PCM and hands it back as normalized float samples in
    /// [-1, 1] — exactly what Overfit's Whisper (<c>WhisperTranscriber.Transcribe</c>) expects, so no
    /// resampling/decoding is needed. Capture runs on a background thread; <see cref="Stop"/> joins it
    /// and returns everything recorded. Everything stays in memory on-device — nothing is written to disk
    /// or sent anywhere.
    /// </summary>
    public sealed class VoiceRecorder
    {
        // Whisper's fixed input rate (MelSpectrogram.SampleRate). Recording straight at 16 kHz avoids a resample.
        public const int SampleRate = 16000;

        private AudioRecord? _record;
        private System.Threading.Thread? _thread;
        private volatile bool _recording;
        private readonly System.Collections.Generic.List<short> _samples = new();

        // Voice-activity tracking for silence-based auto-stop (endpointing).
        private long _totalSamples;
        private long _lastNonSilenceSample;
        private bool _speechDetected;
        private const float SpeechThreshold = 0.020f; // RMS (normalized) to count as "speech started"
        private const float SilenceFloor = 0.012f;    // below this the end-of-speech silence timer runs

        public bool IsRecording => _recording;

        /// <summary>True once the input has crossed the speech level at least once (so we don't endpoint on
        /// leading silence before the user starts talking).</summary>
        public bool SpeechDetected
        {
            get
            {
                lock (_samples)
                {
                    return _speechDetected;
                }
            }
        }

        /// <summary>Seconds of continuous silence at the tail (time since the last above-floor audio).</summary>
        public double SilenceSeconds
        {
            get
            {
                lock (_samples)
                {
                    return (_totalSamples - _lastNonSilenceSample) / (double)SampleRate;
                }
            }
        }

        /// <summary>Number of seconds captured so far (approximate, for a live timer).</summary>
        public double Seconds
        {
            get
            {
                lock (_samples)
                {
                    return _samples.Count / (double)SampleRate;
                }
            }
        }

        public void Start()
        {
            if (_recording)
            {
                return;
            }

            var minBuf = AudioRecord.GetMinBufferSize(SampleRate, ChannelIn.Mono, Encoding.Pcm16bit);
            if (minBuf <= 0)
            {
                minBuf = SampleRate * 2; // ~1 s fallback if the device won't report a size
            }
            var bufSize = Math.Max(minBuf, SampleRate); // headroom so Read() never starves

#pragma warning disable CA1416 // RECORD_AUDIO is declared in the manifest and checked before Start() is called.
            _record = new AudioRecord(AudioSource.Mic, SampleRate, ChannelIn.Mono, Encoding.Pcm16bit, bufSize);
            if (_record.State != State.Initialized)
            {
                _record.Release();
                _record = null;
                throw new InvalidOperationException("Microphone is unavailable.");
            }

            lock (_samples)
            {
                _samples.Clear();
                _totalSamples = 0;
                _lastNonSilenceSample = 0;
                _speechDetected = false;
            }
            _recording = true;
            _record.StartRecording();
#pragma warning restore CA1416

            _thread = new System.Threading.Thread(ReadLoop) { IsBackground = true, Name = "voice-capture" };
            _thread.Start();
        }

        private void ReadLoop()
        {
            var buf = new short[SampleRate / 4]; // ~250 ms chunks
            while (_recording)
            {
                var n = _record!.Read(buf, 0, buf.Length);
                if (n > 0)
                {
                    // RMS of this chunk (normalized to [-1, 1]) drives the silence/speech detection.
                    double sumSq = 0;
                    for (var i = 0; i < n; i++)
                    {
                        var f = buf[i] / 32768f;
                        sumSq += f * f;
                    }
                    var rms = Math.Sqrt(sumSq / n);

                    lock (_samples)
                    {
                        for (var i = 0; i < n; i++)
                        {
                            _samples.Add(buf[i]);
                        }
                        _totalSamples += n;
                        if (rms > SilenceFloor)
                        {
                            _lastNonSilenceSample = _totalSamples;
                        }
                        if (rms > SpeechThreshold)
                        {
                            _speechDetected = true;
                        }
                    }
                }
                else if (n < 0)
                {
                    break; // read error
                }
            }
        }

        /// <summary>Stops recording and returns the captured audio as mono 16 kHz float samples in [-1, 1].</summary>
        public float[] Stop()
        {
            if (!_recording)
            {
                return Array.Empty<float>();
            }

            _recording = false;
            _thread?.Join();
            _thread = null;

            try
            {
                _record?.Stop();
            }
            catch
            {
                // already stopping
            }
            _record?.Release();
            _record = null;

            short[] pcm;
            lock (_samples)
            {
                pcm = _samples.ToArray();
            }

            var f = new float[pcm.Length];
            for (var i = 0; i < pcm.Length; i++)
            {
                f[i] = pcm[i] / 32768f;
            }
            return f;
        }

        /// <summary>Aborts recording and discards the audio.</summary>
        public void Cancel()
        {
            _recording = false;
            try
            {
                _thread?.Join();
            }
            catch
            {
                // ignore
            }
            _thread = null;

            try
            {
                _record?.Stop();
            }
            catch
            {
                // ignore
            }
            _record?.Release();
            _record = null;

            lock (_samples)
            {
                _samples.Clear();
            }
        }
    }
}
