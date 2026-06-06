// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts.Orpheus
{
    /// <summary>
    /// The glue between the Orpheus TTS language model and the <see cref="Snac.Snac"/> codec decoder: turns the
    /// model's flat audio-token stream into SNAC's three hierarchical code levels. Grounded verbatim in the Orpheus
    /// reference (<c>turn_token_into_id</c> / <c>convert_to_audio</c>): each emitted token detokenizes to
    /// <c>&lt;custom_token_N&gt;</c>, whose code is <c>N − 10 − (index%7)·4096</c>; every 7 codes form one frame
    /// that fans out 1 / 2 / 4 codes into levels 0 / 1 / 2 (the 1:2:4 structure of SNAC 24 kHz). Model-free and
    /// exactly testable — an off-by-one here is audible noise, so it is validated independently of the LM.
    /// </summary>
    public static class OrpheusSnacBridge
    {
        /// <summary>Tokens per SNAC frame in Orpheus's flattened layout.</summary>
        public const int FrameStride = 7;

        // Per the reference: code = customTokenNumber - CodeBias - (positionInFrame * CodebookSize).
        private const int CodeBias = 10;
        private const int CodebookSize = 4096;

        private const string CustomTokenPrefix = "<custom_token_";

        /// <summary>
        /// Decodes one audio token: <c>customTokenNumber − 10 − (index%7)·4096</c>, where <paramref name="index"/>
        /// is the position of this token in the accepted audio-token stream (0-based). The <c>index%7</c> term
        /// removes the per-frame-position codebook offset.
        /// </summary>
        public static int DecodeCustomToken(int customTokenNumber, int index)
            => customTokenNumber - CodeBias - ((index % FrameStride) * CodebookSize);

        /// <summary>
        /// The inverse of <see cref="DecodeCustomToken"/>: the <c>&lt;custom_token_N&gt;</c> number that encodes
        /// <paramref name="code"/> at stream position <paramref name="index"/> — <c>code + 10 + (index%7)·4096</c>.
        /// Used when building training targets (voice-clone fine-tuning): the model must learn to emit these.
        /// </summary>
        public static int CustomTokenNumber(int code, int index)
            => code + CodeBias + ((index % FrameStride) * CodebookSize);

        /// <summary>
        /// The inverse of <see cref="Redistribute"/>: flattens SNAC's three levels back into the model's
        /// 7-tokens-per-frame stream (lengths F, 2F, 4F → 7F). Turns an <see cref="Snac.Snac.Encode"/> result into
        /// the audio-token order the LM consumes — the first step of voice-clone dataset prep.
        /// </summary>
        public static int[] Interleave(int[][] levels)
        {
            if (levels.Length != 3)
            {
                throw new OverfitRuntimeException($"SNAC has 3 code levels; got {levels.Length}.");
            }
            var l0 = levels[0];
            var l1 = levels[1];
            var l2 = levels[2];
            var frames = l0.Length;
            if (l1.Length != frames * 2 || l2.Length != frames * 4)
            {
                throw new OverfitRuntimeException(
                    $"SNAC level lengths must be F, 2F, 4F; got {l0.Length}, {l1.Length}, {l2.Length}.");
            }

            var flat = new int[frames * FrameStride];
            for (var j = 0; j < frames; j++)
            {
                var i = FrameStride * j;
                flat[i] = l0[j];
                flat[i + 1] = l1[2 * j];
                flat[i + 4] = l1[(2 * j) + 1];
                flat[i + 2] = l2[4 * j];
                flat[i + 3] = l2[(4 * j) + 1];
                flat[i + 5] = l2[(4 * j) + 2];
                flat[i + 6] = l2[(4 * j) + 3];
            }
            return flat;
        }

        /// <summary>
        /// Extracts the number from the last <c>&lt;custom_token_N&gt;</c> in a detokenized piece (the reference
        /// scans from the right). Returns false if the text holds no complete custom token.
        /// </summary>
        public static bool TryReadCustomTokenNumber(ReadOnlySpan<char> text, out int number)
        {
            number = 0;
            var start = text.LastIndexOf(CustomTokenPrefix);
            if (start < 0)
            {
                return false;
            }
            var rest = text[(start + CustomTokenPrefix.Length)..];
            var end = rest.IndexOf('>');
            if (end <= 0)
            {
                return false;
            }
            return int.TryParse(rest[..end], out number);
        }

        /// <summary>
        /// Fans a flat audio-code stream out into SNAC's three levels (lengths F, 2F, 4F for F complete frames).
        /// Trailing codes that do not complete a frame are dropped. Throws if any code is outside
        /// <c>[0, 4096)</c> (the codebook range) — a sign of a corrupted generation.
        /// </summary>
        public static int[][] Redistribute(ReadOnlySpan<int> audioCodes)
        {
            var frames = audioCodes.Length / FrameStride;
            if (frames == 0)
            {
                throw new OverfitRuntimeException(
                    $"Orpheus produced {audioCodes.Length} audio codes — fewer than one {FrameStride}-token frame.");
            }

            var level0 = new int[frames];
            var level1 = new int[frames * 2];
            var level2 = new int[frames * 4];

            for (var j = 0; j < frames; j++)
            {
                var i = FrameStride * j;
                level0[j] = Validate(audioCodes[i]);
                level1[2 * j] = Validate(audioCodes[i + 1]);
                level1[(2 * j) + 1] = Validate(audioCodes[i + 4]);
                level2[4 * j] = Validate(audioCodes[i + 2]);
                level2[(4 * j) + 1] = Validate(audioCodes[i + 3]);
                level2[(4 * j) + 2] = Validate(audioCodes[i + 5]);
                level2[(4 * j) + 3] = Validate(audioCodes[i + 6]);
            }

            return [level0, level1, level2];
        }

        private static int Validate(int code)
        {
            if ((uint)code >= CodebookSize)
            {
                throw new OverfitRuntimeException(
                    $"Orpheus audio code {code} is outside the codebook range [0, {CodebookSize}).");
            }
            return code;
        }
    }
}
