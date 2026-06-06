// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Tts.Orpheus;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>End-to-end Orpheus TTS on the real model: text → SNAC audio tokens (Orpheus 3B GGUF) → de-interleave
    /// → SNAC decode → 24 kHz waveform, in pure managed .NET. The S3 gate: generation terminates, codes stay in the
    /// valid range (else <see cref="OrpheusVoiceEngine.Synthesize"/> throws), and real intelligible-length audio
    /// comes out. Needs <c>c:\orpheus</c> + <c>c:\snac</c>; [LongFact].</summary>
    public sealed class OrpheusVoiceEngineE2ETests
    {
        [LongFact]
        public void Synthesize_RealModel_ProducesAudio()
        {
            TestModelPaths.Orpheus.RequireGgufPath();
            TestModelPaths.Snac.RequireSafetensorsPath();

            using var engine = OrpheusVoiceEngine.Load(TestModelPaths.Orpheus.GgufPath, TestModelPaths.Snac.Dir);

            var audio = engine.Synthesize("Hi, this is Overfit speaking in pure dot net.", voice: "tara", seed: 1);

            // At least a fraction of a second of audio, all finite and in range.
            Assert.True(audio.Length > engine.SampleRate / 4, $"too little audio: {audio.Length} samples");
            var maxAbs = 0f;
            foreach (var s in audio)
            {
                Assert.True(float.IsFinite(s));
                maxAbs = MathF.Max(maxAbs, MathF.Abs(s));
            }
            Assert.True(maxAbs is > 0.001f and <= 1.0f, $"audio out of range or silent: maxAbs={maxAbs}");

            // Save for manual listening.
            WavWriter.WriteMono(
                Path.Combine(TestModelPaths.Orpheus.Dir, "e2e_tara.wav"),
                audio, engine.SampleRate, WavSampleFormat.Pcm16,
                infoComment: "Overfit Orpheus+SNAC e2e (synthetic)");
        }
    }
}
