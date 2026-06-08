// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Audio.Tts.Orpheus
{
    /// <summary>
    /// Builds the Orpheus TTS input prompt. The model is conditioned on a named preset voice and the text, wrapped
    /// in the audio-start / end-of-turn special tokens the model was trained with — the exact form the GGUF
    /// inference path uses: <c>&lt;|audio|&gt;{voice}: {text}&lt;|eot_id|&gt;</c>.
    /// </summary>
    public static class OrpheusPrompt
    {
        public const string DefaultVoice = "tara";

        /// <summary>The preset voices shipped with <c>orpheus-3b-0.1-ft</c> (best-quality first).</summary>
        public static string[] AvailableVoices { get; } =
            ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"];

        private const string AudioStart = "<|audio|>";
        private const string EndOfTurn = "<|eot_id|>";

        /// <summary>Formats <paramref name="text"/> for <paramref name="voice"/> into the model's prompt string.</summary>
        public static string Format(string text, string voice = DefaultVoice)
            => $"{AudioStart}{voice}: {text}{EndOfTurn}";

        /// <summary>
        /// Builds the <b>canonical</b> Orpheus prompt as raw token ids. The string <see cref="Format"/> form
        /// (<c>&lt;|audio|&gt;…&lt;|eot_id|&gt;</c>) is the LM-Studio shorthand — it relies on the host's chat
        /// template to add the audio-priming control tokens, which our raw-tokenizer path does NOT do, so the model
        /// starts the audio stream unprimed and garbles the first word. The real Orpheus sequence is:
        /// <c>[start_of_human] {voice}: {text} [end_of_text] [end_of_human] [start_of_ai] [start_of_speech]</c>.
        /// The control tokens are the contiguous <c>custom_token</c> block at the audio base (custom_token_0):
        /// start_of_speech = base+1, start_of_human = base+3, end_of_human = base+4, start_of_ai = base+5; the
        /// end_of_text is <c>&lt;|eot_id|&gt;</c> (128009). Returns the full id sequence to feed before decoding audio.
        /// </summary>
        public static int[] BuildPromptTokens(ITokenizer tokenizer, string text, string voice = DefaultVoice)
        {
            var audioBase = EncodeSingle(tokenizer, "<custom_token_0>");
            var beginOfText = EncodeSingle(tokenizer, "<|begin_of_text|>");
            var endOfText = EncodeSingle(tokenizer, EndOfTurn);

            var body = $"{voice}: {text}";
            var bodyIds = new int[tokenizer.CountTokens(body)];
            var n = tokenizer.Encode(body, bodyIds);

            // [start_of_human, begin_of_text] + body + [end_of_text, end_of_human, start_of_ai, start_of_speech].
            // The begin_of_text (BOS) matters: HF's tokenizer auto-prepends it inside prompt_tokens; our Encode does
            // not, and omitting it makes the model emit end_of_speech almost immediately (a ~0.25 s clip).
            var result = new int[2 + n + 4];
            result[0] = audioBase + 3;
            result[1] = beginOfText;
            bodyIds.AsSpan(0, n).CopyTo(result.AsSpan(2));
            result[2 + n + 0] = endOfText;
            result[2 + n + 1] = audioBase + 4;
            result[2 + n + 2] = audioBase + 5;
            result[2 + n + 3] = audioBase + 1;
            return result;
        }

        private static int EncodeSingle(ITokenizer tokenizer, string special)
        {
            Span<int> ids = stackalloc int[8];
            var n = tokenizer.Encode(special, ids);
            if (n != 1)
            {
                throw new OverfitRuntimeException(
                    $"Expected '{special}' to tokenize to one token, got {n} — not an Orpheus tokenizer.");
            }
            return ids[0];
        }

        /// <summary>True if <paramref name="voice"/> is one of the model's preset voices.</summary>
        public static bool IsKnownVoice(string voice)
        {
            foreach (var v in AvailableVoices)
            {
                if (string.Equals(v, voice, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }
            return false;
        }
    }
}
