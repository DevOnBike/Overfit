// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts.Orpheus;
using DevOnBike.Overfit.Audio.Tts.Snac;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>Voice-clone dataset prep against the real Orpheus tokenizer + SNAC encoder: the audio-token base
    /// resolves from the vocab, and a built example's audio tokens decode back to exactly the codes the encoder
    /// produced — so training targets match the generation path. Needs <c>c:\orpheus</c> + <c>c:\snac</c>;
    /// [LongFact].</summary>
    public sealed class VoiceCloneDatasetTests
    {
        [LongFact]
        public void BuildExample_AudioTokens_RoundTripToEncodedCodes()
        {
            TestModelPaths.Orpheus.RequireGgufPath();
            TestModelPaths.Snac.RequireSafetensorsPath();

            var tokenizer = new GgufEmbeddedTokenizer(GgufTokenizer.Load(TestModelPaths.Orpheus.GgufPath));
            var snac = Snac.Load(TestModelPaths.Snac.Dir);
            var builder = new VoiceCloneDatasetBuilder(snac, tokenizer, tokenizer.EndOfTextTokenId);

            // ~0.5 s of 24 kHz audio.
            var audio = new float[12000];
            for (var i = 0; i < audio.Length; i++)
            {
                audio[i] = 0.3f * MathF.Sin((float)(2.0 * Math.PI * 220.0 * i / 24000.0));
            }

            var ex = builder.BuildExample(audio, "this is a test", "tara");
            var codes = snac.Encode(audio);

            // Prompt prefix + audio tokens + end token; audio tokens recover the codes via the decode bridge.
            Assert.True(ex.PromptLength > 0);
            Assert.Equal(tokenizer.EndOfTextTokenId, ex.InputIds[^1]);

            var audioCount = ex.InputIds.Length - ex.PromptLength - 1;
            var decoded = new int[audioCount];
            for (var i = 0; i < audioCount; i++)
            {
                var n = ex.InputIds[ex.PromptLength + i] - builder.AudioTokenBase;
                decoded[i] = OrpheusSnacBridge.DecodeCustomToken(n, i);
            }
            var levels = OrpheusSnacBridge.Redistribute(decoded);

            Assert.Equal(codes[0], levels[0]);
            Assert.Equal(codes[1], levels[1]);
            Assert.Equal(codes[2], levels[2]);
        }
    }
}
