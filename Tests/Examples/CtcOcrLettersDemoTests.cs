// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Training;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Examples
{
    /// <summary>
    /// OCR demo over a real word <b>lexicon</b> (variable-length uppercase words, 5×7 font) showing the
    /// full stack: a <see cref="Crnn"/> trained with CTC, plus an <see cref="NGramCtcLanguageModel"/>
    /// trained on the same lexicon used for LM-rescored beam decoding. Reports greedy vs LM-rescored
    /// recognition head-to-head — the n-gram LM steers the CRNN's confusable-glyph misreads back toward
    /// real words, so LM-beam accuracy is ≥ greedy.
    /// </summary>
    public sealed class CtcOcrLettersDemoTests
    {
        private const int H = 7;
        private const int GlyphW = 5;
        private const int Wmax = 32;      // ≥ 5 letters × 6 cols
        private const int LetterClasses = 26;   // A–Z (the LM's class count)
        private const int Classes = 27;          // + blank (class 26, the Crnn default)

        private static readonly string[] Lexicon =
        [
            "CAT", "DOG", "SUN", "FOX", "OWL",
            "MOON", "STAR", "TREE", "FISH", "BIRD", "BLUE", "WOLF", "BEAR", "DEER", "HAWK", "LEAF",
            "GREEN", "RIVER", "OCEAN", "CLOUD", "STORM", "LIGHT", "NIGHT", "STONE", "FLAME", "FROST",
        ];

        // 5×7 uppercase font A–Z.
        private static readonly string[][] Font =
        [
            ["..#..", ".#.#.", "#...#", "#...#", "#####", "#...#", "#...#"], // A
            ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."], // B
            [".####", "#....", "#....", "#....", "#....", "#....", ".####"], // C
            ["###..", "#..#.", "#...#", "#...#", "#...#", "#..#.", "###.."], // D
            ["#####", "#....", "#....", "####.", "#....", "#....", "#####"], // E
            ["#####", "#....", "#....", "####.", "#....", "#....", "#...."], // F
            [".####", "#....", "#....", "#..##", "#...#", "#...#", ".####"], // G
            ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"], // H
            ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"], // I
            ["..###", "...#.", "...#.", "...#.", "#..#.", "#..#.", ".##.."], // J
            ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"], // K
            ["#....", "#....", "#....", "#....", "#....", "#....", "#####"], // L
            ["#...#", "##.##", "#.#.#", "#...#", "#...#", "#...#", "#...#"], // M
            ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"], // N
            [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."], // O
            ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."], // P
            [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"], // Q
            ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"], // R
            [".####", "#....", "#....", ".###.", "....#", "....#", "####."], // S
            ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."], // T
            ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."], // U
            ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."], // V
            ["#...#", "#...#", "#...#", "#...#", "#.#.#", "##.##", "#...#"], // W
            ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"], // X
            ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."], // Y
            ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"], // Z
        ];

        private readonly ITestOutputHelper _out;
        public CtcOcrLettersDemoTests(ITestOutputHelper output) => _out = output;

        [LongFact]  // runtime unmeasured — the test failed after 8s (2026-08-07)
        [Trait("Category", "Demo")]
        public void RecognisesLexiconWords_WithNGramLanguageModel()
        {
            const int optSteps = 400;
            const int accumWords = 8;
            const float lrMax = 0.01f, lrMin = 1e-4f;

            // SEED THE WEIGHTS, not just the data order. The `new Random(20260527)` below seeds which
            // words this test draws; it does NOT touch weight initialisation, which is what actually made
            // this test flaky. Measured 2026-08-07 on unchanged code: 5/24 recognised on one run, 23/24 on
            // another. The LSTM inside the CRNN drew from `Random.Shared`, outside any seed's reach, until
            // it was routed through MathUtils the same day.
            MathUtils.SetSeed(20260527);

            using var ocr = new Crnn(
                imageHeight: H, imageWidth: Wmax, classCount: Classes,
                convChannels: 32, kernelSize: 3, lstmHidden: 128);
            ocr.Train();

            // Train a trigram LM on the lexicon — it learns which letter sequences are real words.
            var lm = new NGramCtcLanguageModel(classCount: LetterClasses, order: 3, smoothing: 0.05);
            foreach (var word in Lexicon)
            {
                lm.Train(Labels(word));
            }

            using var optimizer = new Adam(ocr.Parameters(), lrMax);
            using var graph = new ComputationGraph(24_000_000);
            var rng = new Random(20260527);

            var firstLoss = 0f;
            var tailSum = 0f;
            var tailCount = 0;

            // Non-finite CTC losses used to be swallowed silently by the `continue` below: no count, no
            // report, no assertion. When every word in a batch produced one, `optimizer.Step()` still ran —
            // on gradients that had just been zeroed — so training quietly stopped advancing and the only
            // symptom was the recognition assertion at the end failing with "5/24 too low". That says the
            // model did not learn; it does not say why, and this counter is what tells the difference
            // between a diverged run and a decoder problem.
            var nonFiniteLosses = 0;

            for (var step = 0; step < optSteps; step++)
            {
                optimizer.LearningRate = LearningRateSchedule.Cosine(step, optSteps, lrMax, lrMin);
                optimizer.ZeroGrad();

                var batchLoss = 0f;
                for (var k = 0; k < accumWords; k++)
                {
                    var label = Labels(Lexicon[rng.Next(Lexicon.Length)]);
                    var image = Render(label);

                    graph.Reset();
                    ocr.InvalidateParameterCaches();
                    using var input = ocr.CreateInput(image);
                    var logits = ocr.Forward(graph, input);
                    var loss = ocr.ComputeCtcLoss(logits, label);
                    if (!float.IsFinite(loss))
                    {
                        nonFiniteLosses++;

                        continue;
                    }

                    graph.BackwardFromGrad(logits);
                    batchLoss += loss;
                }
                optimizer.Step();

                batchLoss /= accumWords;
                if (step == 0)
                {
                    firstLoss = batchLoss;
                }
                if (step >= optSteps - 50)
                {
                    tailSum += batchLoss;
                    tailCount++;
                }
                if (step == 0 || (step + 1) % 100 == 0)
                {
                    _out.WriteLine($"step {step + 1,4}/{optSteps}  loss={batchLoss:F4}");
                }
            }

            var tailLoss = tailSum / tailCount;
            _out.WriteLine($"loss: {firstLoss:F4} -> tail-avg {tailLoss:F4}   "
                           + $"non-finite losses {nonFiniteLosses}/{optSteps * accumWords}");

            // TRAINING FIRST, RECOGNITION SECOND — the order is the point. Measured 2026-08-07, on the first
            // run this test ever had, it failed with "LM-beam recognition 5/24 too low", which is the
            // downstream symptom of a network that did not learn and says nothing about the cause. Whoever
            // reads that has to reconstruct by hand whether training diverged, whether the CRNN is fine and
            // the beam search is wrong, or whether the n-gram model is at fault. These two assertions answer
            // the first question before the third one is asked.
            Assert.True(nonFiniteLosses == 0,
                $"{nonFiniteLosses} of {optSteps * accumWords} CTC losses were not finite, so those steps "
                + "contributed no gradient. Training did not run as written; any recognition number below "
                + "is meaningless.");

            // 0.5x is NOT a measured threshold and should not be read as one — it is a collapse detector,
            // chosen loose on purpose. The weight initialisation here is not seedable (MathUtils seeds from
            // a GUID, LSTMCell draws from Random.Shared), so a run-to-run spread exists that nobody has
            // quantified; a tight bound would turn this into a second flaky assertion instead of a
            // diagnostic. Tighten it to a real number once the initialisation takes a seed.
            Assert.True(tailLoss < firstLoss * 0.5f,
                $"CTC loss barely moved: {firstLoss:F4} -> {tailLoss:F4} over {optSteps} steps. The network "
                + "did not train, so the recognition assertions below would only be measuring noise.");

            // ── Greedy vs LM-rescored beam over the lexicon ──
            ocr.Eval();
            var greedyOk = 0;
            var lmOk = 0;
            const int trials = 24;
            for (var i = 0; i < trials; i++)
            {
                var word = Lexicon[rng.Next(Lexicon.Length)];
                var label = Labels(word);
                var image = Render(label);

                var greedy = ocr.Recognize(graph, image);                                  // best path
                var lmBeam = ocr.Recognize(graph, image, beamWidth: 12, lm, languageModelWeight: 1.0); // LM-rescored

                var greedyHit = Equal(greedy, label);
                var lmHit = Equal(lmBeam, label);
                if (greedyHit)
                {
                    greedyOk++;
                }
                if (lmHit)
                {
                    lmOk++;
                }

                if (greedyHit != lmHit)
                {
                    _out.WriteLine($"  {word}: greedy={Text(greedy)}{(greedyHit ? "" : " x")}  " +
                                   $"lm={Text(lmBeam)}{(lmHit ? " (LM fixed)" : " x")}");
                }
            }

            _out.WriteLine($"greedy {greedyOk}/{trials}   LM-beam {lmOk}/{trials}");
            Assert.True(lmOk >= greedyOk, $"LM-beam ({lmOk}) should not be worse than greedy ({greedyOk}).");
            Assert.True(lmOk >= trials - 2, $"LM-beam recognition {lmOk}/{trials} too low.");
        }

        private static int[] Labels(string word)
        {
            var labels = new int[word.Length];
            for (var i = 0; i < word.Length; i++)
            {
                labels[i] = word[i] - 'A';
            }
            return labels;
        }

        private static string Text(int[] labels)
        {
            var sb = new StringBuilder();
            foreach (var l in labels)
            {
                sb.Append((char)('A' + l));
            }
            return sb.ToString();
        }

        private static bool Equal(int[] a, int[] b)
        {
            if (a.Length != b.Length)
            {
                return false;
            }
            for (var i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i])
                {
                    return false;
                }
            }
            return true;
        }

        // Row-major [H, Wmax] image, letters left-aligned, blank-separated, zero-padded.
        private static float[] Render(int[] label)
        {
            var image = new float[H * Wmax];
            var col = 0;
            for (var d = 0; d < label.Length; d++)
            {
                var glyph = Font[label[d]];
                for (var gc = 0; gc < GlyphW; gc++)
                {
                    for (var r = 0; r < H; r++)
                    {
                        if (glyph[r][gc] == '#')
                        {
                            image[r * Wmax + col] = 1f;
                        }
                    }
                    col++;
                }
                col++; // blank separator column
            }
            return image;
        }
    }
}
