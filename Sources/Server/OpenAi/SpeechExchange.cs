// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Tts;
using DevOnBike.Overfit.Audio.Tts.Orpheus;

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>
    /// The one implementation of <c>POST /v1/audio/speech</c>, shared by every host: validates the request,
    /// synthesizes with the single <see cref="OrpheusVoiceEngine"/> (caller serializes access to it) and
    /// writes WAV or raw PCM-16 through an <see cref="IOpenAiResponseSink"/>. The WAV/PCM encoders live here
    /// so neither host duplicates the audio framing.
    /// </summary>
    public static class SpeechExchange
    {
        public static void Handle(SpeechRequest? req, OrpheusVoiceEngine tts, IOpenAiResponseSink sink)
        {
            ArgumentNullException.ThrowIfNull(tts);
            ArgumentNullException.ThrowIfNull(sink);

            if (req == null || string.IsNullOrWhiteSpace(req.Input))
            {
                WriteError(sink, 400, "'input' is required.");
                return;
            }

            var format = (req.ResponseFormat ?? "wav").ToLowerInvariant();
            if (format is not ("wav" or "pcm"))
            {
                WriteError(sink, 400, $"response_format '{req.ResponseFormat}' is not supported; use 'wav' or 'pcm'.");
                return;
            }

            var voice = string.IsNullOrWhiteSpace(req.Voice) ? OrpheusPrompt.DefaultVoice : req.Voice!;
            var audio = tts.Synthesize(req.Input!, voice);

            var isPcm = format == "pcm";

            // Raw PCM has no container, so the marker cannot travel inside the payload. It travels on the
            // media type instead, which is a legitimate place for parameters and reaches the client in the
            // Content-Type header. This response is generated speech either way; the format a caller asked
            // for must not decide whether it says so.
            var contentType = isPcm
                ? "audio/pcm; synthetic=true; generated-by=Overfit"
                : "audio/wav";
            var bytes = isPcm ? ToPcm16Bytes(audio) : ToWavBytes(audio, tts.SampleRate, voice);

            sink.WriteBinary(200, contentType, bytes);
        }

        private static byte[] ToWavBytes(float[] audio, int sampleRate, string voice)
        {
            using var ms = new MemoryStream();
            WavWriter.WriteMono(ms, audio, sampleRate, WavSampleFormat.Pcm16,
                SyntheticSpeechMetadata.ForNow(voice).ToInfoComment());
            return ms.ToArray();
        }

        private static byte[] ToPcm16Bytes(float[] samples)
        {
            var bytes = new byte[samples.Length * 2];
            for (var i = 0; i < samples.Length; i++)
            {
                var clamped = Math.Clamp(samples[i], -1f, 1f);
                var v = (short)MathF.Round(clamped * 32767f);
                bytes[i * 2] = (byte)(v & 0xFF);
                bytes[(i * 2) + 1] = (byte)((v >> 8) & 0xFF);
            }
            return bytes;
        }

        private static void WriteError(IOpenAiResponseSink sink, int status, string message)
        {
            var body = new OpenAiErrorResponse { Error = new OpenAiError { Message = message } };
            sink.WriteBody(status, "application/json",
                JsonSerializer.Serialize(body, OpenAiJsonContext.Default.OpenAiErrorResponse));
        }
    }
}
