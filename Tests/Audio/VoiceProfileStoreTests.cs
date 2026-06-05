// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>Voice-profile persistence (the on-disk side of <c>overfit voice enroll</c>) round-trips, including
    /// the cloned-voice speaker embedding, with a reflection-free manifest format.</summary>
    public sealed class VoiceProfileStoreTests : IDisposable
    {
        private readonly string _dir = Path.Combine(Path.GetTempPath(), "overfit-voices-" + Guid.NewGuid().ToString("N"));

        public void Dispose()
        {
            try
            {
                if (Directory.Exists(_dir))
                {
                    Directory.Delete(_dir, recursive: true);
                }
            }
            catch
            {
                // best-effort cleanup
            }
        }

        [Fact]
        public void PresetVoice_SaveLoad_RoundTrips()
        {
            var profile = new VoiceProfile("maciej", "pl", speakerEmbedding: null, referenceAudioPath: @"C:\samples\maciej.wav");
            VoiceProfileStore.Save(profile, _dir);

            var loaded = VoiceProfileStore.Load("maciej", _dir);

            Assert.Equal("maciej", loaded.Id);
            Assert.Equal("pl", loaded.Language);
            Assert.False(loaded.IsCloned);
            Assert.Equal(@"C:\samples\maciej.wav", loaded.ReferenceAudioPath);
        }

        [Fact]
        public void ClonedVoice_EmbeddingRoundTrips()
        {
            float[] embedding = [0.1f, -0.2f, 0.3f, 0.4f, -0.5f];
            VoiceProfileStore.Save(new VoiceProfile("anna", "pl", embedding), _dir);

            var loaded = VoiceProfileStore.Load("anna", _dir);

            Assert.True(loaded.IsCloned);
            Assert.Equal(embedding, loaded.SpeakerEmbedding);
        }

        [Fact]
        public void List_ReturnsEnrolledIds()
        {
            VoiceProfileStore.Save(VoiceProfile.Preset("alpha", "en"), _dir);
            VoiceProfileStore.Save(VoiceProfile.Preset("beta", "pl"), _dir);

            var ids = VoiceProfileStore.List(_dir);

            Assert.Contains("alpha", ids);
            Assert.Contains("beta", ids);
        }

        [Fact]
        public void Load_Missing_Throws()
        {
            Assert.Throws<OverfitFormatException>(() => VoiceProfileStore.Load("nope", _dir));
        }
    }
}
