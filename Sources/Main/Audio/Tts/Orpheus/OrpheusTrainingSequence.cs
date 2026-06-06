// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts.Orpheus
{
    /// <summary>
    /// Builds an Orpheus voice-clone training example from a tokenized prompt and the SNAC codes of the matching
    /// audio. The target sequence mirrors the <i>generation</i> path exactly — prompt
    /// (<c>&lt;|audio|&gt;{voice}: {text}&lt;|eot_id|&gt;</c>) followed by the flattened audio-token stream and an
    /// end-of-speech token — so what the model learns is precisely what <see cref="OrpheusVoiceEngine"/> later
    /// generates. Model-free (token-id arithmetic); the audio-token id of a code is
    /// <c>audioTokenBase + code + 10 + (index%7)·4096</c>.
    /// </summary>
    public static class OrpheusTrainingSequence
    {
        /// <summary>
        /// Assembles the training example. <paramref name="promptTokenIds"/> is the tokenized prompt;
        /// <paramref name="codes"/> are the SNAC levels from <see cref="Snac.Snac.Encode"/>;
        /// <paramref name="audioTokenBase"/> is the vocab id of <c>&lt;custom_token_0&gt;</c> (resolve once from the
        /// tokenizer); <paramref name="endOfSpeechTokenId"/> terminates the audio stream.
        /// </summary>
        public static OrpheusTrainingExample Build(
            int[] promptTokenIds, int[][] codes, int audioTokenBase, int endOfSpeechTokenId)
        {
            var flat = OrpheusSnacBridge.Interleave(codes);

            var inputIds = new int[promptTokenIds.Length + flat.Length + 1];
            promptTokenIds.AsSpan().CopyTo(inputIds);

            for (var i = 0; i < flat.Length; i++)
            {
                inputIds[promptTokenIds.Length + i] = audioTokenBase + OrpheusSnacBridge.CustomTokenNumber(flat[i], i);
            }
            inputIds[^1] = endOfSpeechTokenId;

            return new OrpheusTrainingExample(inputIds, promptTokenIds.Length);
        }
    }
}
