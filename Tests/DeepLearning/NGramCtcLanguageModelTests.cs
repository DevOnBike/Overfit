
// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.Tests.DeepLearning
{
    /// <summary>
    /// Tests for the ready-made <see cref="NGramCtcLanguageModel"/>: it learns a conditional
    /// distribution from a corpus, smooths unseen continuations to a finite (low) score, and — wired into
    /// <see cref="CtcDecoder.BeamSearchDecode(ReadOnlySpan{float}, int, int, int, int, ICtcLanguageModel, double, double)"/> —
    /// steers an ambiguous decode toward the pattern it was trained on.
    /// </summary>
    public sealed class NGramCtcLanguageModelTests
    {
        [Fact]
        public void LearnsConditionalDistribution_AndSmoothsUnseen()
        {
            // After label 0 the corpus always has label 1, never 2.
            var lm = new NGramCtcLanguageModel(classCount: 4, order: 2, smoothing: 0.1);
            lm.Train(new[] { 0, 1, 0, 1, 0, 1 });

            var pSeen = lm.LogProbability([0], 1);     // 0 → 1 (frequent)
            var pUnseen = lm.LogProbability([0], 2);   // 0 → 2 (never observed)

            Assert.True(pSeen > pUnseen, $"expected P(1|0) > P(2|0), got {pSeen} vs {pUnseen}");
            Assert.True(double.IsFinite(pUnseen), "smoothing must keep unseen continuations finite");
        }

        [Fact]
        public void TrainedNGram_SteersBeamSearch()
        {
            // Classes A=0, B=1, C=2, blank=3. Train a bigram that A is always followed by C.
            var lm = new NGramCtcLanguageModel(classCount: 4, order: 2, smoothing: 0.01);
            for (var i = 0; i < 20; i++) { lm.Train(new[] { 0, 2 }); }

            // Acoustics: step0 forces A; step1 prefers B (2.0) over C (1.5).
            const int t = 2, classes = 4, blank = 3;
            var logits = new float[t * classes];
            logits[0] = 10f;
            logits[classes + 1] = 2.0f;   // B
            logits[classes + 2] = 1.5f;   // C

            var noLm = CtcDecoder.BeamSearchDecode(logits, t, classes, blank, beamWidth: 8);
            var withLm = CtcDecoder.BeamSearchDecode(logits, t, classes, blank, beamWidth: 8, lm, languageModelWeight: 3.0);

            Assert.Equal(new[] { 0, 1 }, noLm);     // acoustic → "AB"
            Assert.Equal(new[] { 0, 2 }, withLm);   // n-gram (A→C) steers → "AC"
        }

        [Fact]
        public void UntrainedModel_IsUniform()
        {
            var lm = new NGramCtcLanguageModel(classCount: 5, order: 3, smoothing: 0.5);
            // No training ⇒ every label equally likely (≈ log 1/5) for any prefix.
            var p0 = lm.LogProbability([1, 2], 0);
            var p3 = lm.LogProbability([1, 2], 3);
            Assert.Equal(p0, p3, 6);
            Assert.Equal(Math.Log(1.0 / 5), p0, 6);
        }
    }
}
