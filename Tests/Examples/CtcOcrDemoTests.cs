// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.Linq;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Examples
{
    /// <summary>
    /// End-to-end OCR demo driving the reusable <see cref="Crnn"/> facade: three numbers to build the
    /// model (<c>new Crnn(height, width, classCount)</c>), <see cref="Crnn.Forward"/> +
    /// <see cref="Crnn.ComputeCtcLoss"/> to train, <see cref="Crnn.Recognize(ComputationGraph, ReadOnlySpan{float})"/>
    /// to read text back. Renders 2-3 digit strings with a 3×5 bitmap font into a fixed-width image; the
    /// CRNN (conv → map-to-sequence → LSTM → linear) learns to read them via CTC with no image↔label
    /// alignment.
    /// </summary>
    public sealed class CtcOcrDemoTests
    {
        private const int H = 5;          // image height (= glyph height)
        private const int GlyphW = 3;     // glyph width in columns
        private const int Wmax = 13;      // fixed image width (longest word + padding)
        private const int Classes = 11;   // digits 0-9 + blank (= class 10, the Crnn default)

        private static readonly string[][] Font =
        [
            ["###", "#.#", "#.#", "#.#", "###"], ["...", ".#.", ".#.", ".#.", "..."],
            ["###", "..#", "###", "#..", "###"], ["###", "..#", "###", "..#", "###"],
            ["#.#", "#.#", "###", "..#", "..#"], ["###", "#..", "###", "..#", "###"],
            ["###", "#..", "###", "#.#", "###"], ["###", "..#", "..#", ".#.", ".#."],
            ["###", "#.#", "###", "#.#", "###"], ["###", "#.#", "###", "..#", "###"],
        ];

        private readonly ITestOutputHelper _out;
        public CtcOcrDemoTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        [Trait("Category", "Demo")]
        public void TrainsAndRecognizesSyntheticDigitStrings()
        {
            const int steps = 600;

            // Minimal config: image size + class count. Conv channels / kernel / LSTM size default.
            using var ocr = new Crnn(imageHeight: H, imageWidth: Wmax, classCount: Classes);
            ocr.Train();

            using var optimizer = new Adam(ocr.Parameters(), 0.01f);
            using var graph = new ComputationGraph(16_000_000);
            var rng = new Random(20260527);

            var firstLoss = 0f;
            var lastLoss = 0f;
            var converged = -1;

            for (var step = 0; step < steps; step++)
            {
                var digits = RandomWord(rng, 2, 3);
                var image = Render(digits);

                graph.Reset();
                ocr.InvalidateParameterCaches();

                using var input = ocr.CreateInput(image);
                var logits = ocr.Forward(graph, input);
                var loss = ocr.ComputeCtcLoss(logits, digits);
                if (!float.IsFinite(loss)) { continue; }

                optimizer.ZeroGrad();
                graph.BackwardFromGrad(logits);
                optimizer.Step();

                if (step == 0) { firstLoss = loss; }
                lastLoss = loss;
                if (converged < 0 && loss < 0.1f) { converged = step + 1; }

                if (step == 0 || (step + 1) % 100 == 0)
                {
                    _out.WriteLine($"step {step + 1,4}/{steps}  loss={loss:F4}");
                }
            }

            _out.WriteLine($"loss: {firstLoss:F4} -> {lastLoss:F4}   first<0.1 at step {converged}");
            Assert.True(lastLoss < firstLoss * 0.25f, $"loss did not converge: {firstLoss:F4} -> {lastLoss:F4}");

            // ── Recognition on fresh words via the facade ──
            ocr.Eval();
            var correct = 0;
            const int trials = 8;
            for (var i = 0; i < trials; i++)
            {
                var digits = RandomWord(rng, 2, 3);
                var image = Render(digits);
                var decoded = ocr.Recognize(graph, image);

                var ok = decoded.SequenceEqual(digits);
                if (ok) { correct++; }
                _out.WriteLine($"target=[{string.Join("", digits)}] decoded=[{string.Join("", decoded)}] {(ok ? "OK" : "x")}");
            }

            _out.WriteLine($"recognised {correct}/{trials}");
            Assert.True(correct >= trials - 2, $"OCR recognised only {correct}/{trials} words.");
        }

        private static int[] RandomWord(Random rng, int minLen, int maxLen)
        {
            var len = rng.Next(minLen, maxLen + 1);
            var word = new int[len];
            for (var i = 0; i < len; i++) { word[i] = rng.Next(0, 10); }
            return word;
        }

        // Row-major [H, Wmax] image, digits left-aligned, blank-separated, zero-padded.
        private static float[] Render(int[] digits)
        {
            var image = new float[H * Wmax];
            var col = 0;
            for (var d = 0; d < digits.Length; d++)
            {
                var glyph = Font[digits[d]];
                for (var gc = 0; gc < GlyphW; gc++)
                {
                    for (var r = 0; r < H; r++)
                    {
                        if (glyph[r][gc] == '#') { image[r * Wmax + col] = 1f; }
                    }
                    col++;
                }
                col++; // blank separator column
            }
            return image;
        }
    }
}
