// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.Tests.DeepLearning
{
    /// <summary>Greedy CTC decode: collapse repeats, drop blanks, keep blank-separated repeats.</summary>
    public sealed class CtcDecoderTests
    {
        private const int C = 3;
        private const int Blank = 2;

        // Sharp one-hot logits whose per-timestep argmax is the given class sequence (the chosen class
        // gets a large logit so softmax ≈ 1 — makes the best path also the best labeling).
        private static float[] FromArgmax(params int[] classes)
        {
            var logits = new float[classes.Length * C];
            for (var t = 0; t < classes.Length; t++) { logits[t * C + classes[t]] = 12f; }
            return logits;
        }

        [Fact]
        public void CollapsesRepeats_DropsBlanks_KeepsBlankSeparatedRepeats()
        {
            // argmax path: 0 0 _ 0 1 _ 1  ⇒  0 (merge), 0 (after blank), 1, 1 (after blank)
            var logits = FromArgmax(0, 0, Blank, 0, 1, Blank, 1);
            var decoded = CtcDecoder.GreedyDecode(logits, 7, C, Blank);
            Assert.Equal(new[] { 0, 0, 1, 1 }, decoded);
        }

        [Fact]
        public void AdjacentDuplicate_WithoutBlank_EmitsOnce()
        {
            var logits = FromArgmax(1, 1, 1);
            var decoded = CtcDecoder.GreedyDecode(logits, 3, C, Blank);
            Assert.Equal(new[] { 1 }, decoded);
        }

        [Fact]
        public void AllBlank_DecodesEmpty()
        {
            var logits = FromArgmax(Blank, Blank, Blank);
            var decoded = CtcDecoder.GreedyDecode(logits, 3, C, Blank);
            Assert.Empty(decoded);
        }

        [Fact]
        public void BeamSearch_BeatsGreedy_WhenBestLabelingIsNotBestPath()
        {
            // Textbook case (Hannun 2014): 2 timesteps, classes {A=0, blank=1}, p(A)=0.4, p(blank)=0.6.
            // Greedy picks blank,blank → "" (path prob 0.36). But "A" sums its alignments
            // (A,blank)+(blank,A)+(A,A) = 0.24+0.24+0.16 = 0.64 > 0.36, so it is the true best labeling.
            const int t = 2, classes = 2, blank = 1;
            var logits = new float[t * classes];
            for (var i = 0; i < t; i++)
            {
                logits[i * classes + 0] = MathF.Log(0.4f); // A
                logits[i * classes + 1] = MathF.Log(0.6f); // blank
            }

            var greedy = CtcDecoder.GreedyDecode(logits, t, classes, blank);
            var beam = CtcDecoder.BeamSearchDecode(logits, t, classes, blank, beamWidth: 8);

            Assert.Empty(greedy);                  // greedy → ""
            Assert.Equal(new[] { 0 }, beam);       // beam → "A"
        }

        [Fact]
        public void BeamSearch_MatchesGreedy_OnConfidentPath()
        {
            // When one path dominates, the best labeling == the best path.
            var logits = FromArgmax(0, 0, Blank, 0, 1, Blank, 1);
            // Sharpen so the argmax path overwhelmingly dominates (one-hot is already 1 vs 0).
            var greedy = CtcDecoder.GreedyDecode(logits, 7, C, Blank);
            var beam = CtcDecoder.BeamSearchDecode(logits, 7, C, Blank, beamWidth: 8);
            Assert.Equal(greedy, beam);
        }

        // After the label A (0), strongly prefer C (2) over B (1).
        private sealed class FavorCAfterA : ICtcLanguageModel
        {
            public double LogProbability(ReadOnlySpan<int> prefix, int nextLabel)
            {
                if (prefix.Length > 0 && prefix[^1] == 0)
                {
                    if (nextLabel == 2) { return 0.0; }     // log 1
                    if (nextLabel == 1) { return -20.0; }   // ~forbidden
                }
                return -1.0;
            }
        }

        [Fact]
        public void BeamSearch_LanguageModel_OverridesAcousticPreference()
        {
            // classes A=0, B=1, C=2, blank=3. Step0 forces A; step1 acoustically prefers B over C.
            const int t = 2, classes = 4, blank = 3;
            var logits = new float[t * classes];
            logits[0] = 10f;                       // step0: A
            logits[classes + 1] = 2.0f;            // step1: B (higher)
            logits[classes + 2] = 1.5f;            // step1: C (lower)

            var noLm = CtcDecoder.BeamSearchDecode(logits, t, classes, blank, beamWidth: 8);
            var withLm = CtcDecoder.BeamSearchDecode(
                logits, t, classes, blank, beamWidth: 8, new FavorCAfterA(), languageModelWeight: 2.0);

            Assert.Equal(new[] { 0, 1 }, noLm);    // acoustic → "AB"
            Assert.Equal(new[] { 0, 2 }, withLm);  // LM steers → "AC"
        }
    }
}
