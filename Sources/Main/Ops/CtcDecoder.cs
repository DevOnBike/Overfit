// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Ops
{
    /// <summary>
    /// Decoding for <see cref="CtcLoss"/>-trained models — turns per-timestep class scores back into a
    /// label sequence. Two strategies:
    /// <list type="bullet">
    ///   <item><see cref="GreedyDecode"/> — best <b>path</b>: argmax per timestep then collapse. Fast,
    ///         allocation-light, but the single best path can disagree with the most probable labeling.</item>
    ///   <item><c>BeamSearchDecode</c> — best <b>labeling</b>: CTC prefix beam search (Hannun et
    ///         al. 2014) sums alignment probabilities per candidate string, so it recovers labelings whose
    ///         probability mass is spread across several alignments (and is the natural hook for a
    ///         language model). Usually a small accuracy gain over greedy on ambiguous input.</item>
    /// </list>
    /// </summary>
    public static class CtcDecoder
    {
        private const double NegInf = double.NegativeInfinity;

        /// <summary>
        /// Greedy/best-path decode of row-major <c>[timeSteps × classCount]</c> logits (or any monotonic
        /// transform — only the per-timestep argmax matters) into a label sequence, collapsing repeats
        /// and removing the blank.
        /// </summary>
        public static int[] GreedyDecode(
            ReadOnlySpan<float> logits, int timeSteps, int classCount, int blankIndex)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(timeSteps);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(classCount);
            ArgumentOutOfRangeException.ThrowIfNegative(blankIndex);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(blankIndex, classCount);
            if (logits.Length < checked(timeSteps * classCount))
            {
                throw new ArgumentException("logits length < timeSteps*classCount.", nameof(logits));
            }

            var output = new List<int>();
            var previous = -1;
            for (var t = 0; t < timeSteps; t++)
            {
                var baseT = t * classCount;
                var best = 0;
                var bestVal = logits[baseT];
                for (var k = 1; k < classCount; k++)
                {
                    var v = logits[baseT + k];
                    if (v > bestVal) { bestVal = v; best = k; }
                }

                // CTC collapse: emit only when the class changes and is not the blank. A repeated label
                // must be separated by a blank (or a different class) to be emitted twice.
                if (best != previous && best != blankIndex)
                {
                    output.Add(best);
                }
                previous = best;
            }

            return output.ToArray();
        }

        /// <summary>
        /// CTC prefix beam search over row-major <c>[timeSteps × classCount]</c> logits. Returns the
        /// highest-probability <b>labeling</b> (summing over its alignments), keeping the top
        /// <paramref name="beamWidth"/> prefixes at each step. The log-softmax is taken internally; all
        /// arithmetic is in log-space. <paramref name="beamWidth"/> = 1 approaches greedy.
        /// </summary>
        public static int[] BeamSearchDecode(
            ReadOnlySpan<float> logits, int timeSteps, int classCount, int blankIndex, int beamWidth = 16)
            => BeamSearchCore(logits, timeSteps, classCount, blankIndex, beamWidth, languageModel: null, 0.0, 0.0);

        /// <summary>
        /// CTC prefix beam search with <b>language-model rescoring</b> (Hannun et al. 2014). Each time a
        /// beam's labeling grows by a label <c>c</c>, its log-prob gains
        /// <paramref name="languageModelWeight"/> · <c>languageModel.LogProbability(prefix, c)</c>; beams
        /// are ranked by <c>logProb + insertionBonus · length</c> (the bonus offsets the LM's bias toward
        /// shorter strings). Lets a char/word LM steer decoding toward plausible text.
        /// </summary>
        public static int[] BeamSearchDecode(
            ReadOnlySpan<float> logits, int timeSteps, int classCount, int blankIndex, int beamWidth,
            ICtcLanguageModel languageModel, double languageModelWeight = 1.0, double insertionBonus = 0.0)
        {
            ArgumentNullException.ThrowIfNull(languageModel);
            return BeamSearchCore(logits, timeSteps, classCount, blankIndex, beamWidth, languageModel, languageModelWeight, insertionBonus);
        }

        private static int[] BeamSearchCore(
            ReadOnlySpan<float> logits, int timeSteps, int classCount, int blankIndex, int beamWidth,
            ICtcLanguageModel? languageModel, double languageModelWeight, double insertionBonus)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(timeSteps);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(classCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(beamWidth);
            ArgumentOutOfRangeException.ThrowIfNegative(blankIndex);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(blankIndex, classCount);
            if (logits.Length < checked(timeSteps * classCount))
            {
                throw new ArgumentException("logits length < timeSteps*classCount.", nameof(logits));
            }

            var logp = LogSoftmaxPerTimestep(logits, timeSteps, classCount);

            // Each beam tracks log-prob of the prefix ending in a blank (Pb) vs a non-blank (Pnb).
            var beam = new Dictionary<string, Beam>
            {
                [""] = new Beam([], logPb: 0.0, logPnb: NegInf),
            };

            for (var t = 0; t < timeSteps; t++)
            {
                var baseT = t * classCount;
                var next = new Dictionary<string, Beam>();

                foreach (var (key, e) in beam)
                {
                    var pTotal = LogSumExp(e.LogPb, e.LogPnb);

                    // (a) extend by blank → same prefix, blank-ending.
                    var same = GetOrAdd(next, key, e.Labels);
                    same.LogPb = LogSumExp(same.LogPb, pTotal + logp[baseT + blankIndex]);

                    // (b) repeat the last label with no blank between → same prefix, non-blank-ending.
                    if (e.Labels.Length > 0)
                    {
                        var lastLabel = e.Labels[^1];
                        same.LogPnb = LogSumExp(same.LogPnb, e.LogPnb + logp[baseT + lastLabel]);
                    }

                    // (c) extend by each non-blank class.
                    for (var c = 0; c < classCount; c++)
                    {
                        if (c == blankIndex) { continue; }

                        var lp = logp[baseT + c];
                        // A repeat of the last label can only follow a blank (else it would merge).
                        var contribution = (e.Labels.Length > 0 && c == e.Labels[^1])
                            ? e.LogPb + lp
                            : pTotal + lp;

                        // The labeling grows by c here → apply the language-model score.
                        if (languageModel is not null)
                        {
                            contribution += languageModelWeight * languageModel.LogProbability(e.Labels, c);
                        }

                        var extended = GetOrAdd(next, key + (char)c, Append(e.Labels, c));
                        extended.LogPnb = LogSumExp(extended.LogPnb, contribution);
                    }
                }

                beam = Prune(next, beamWidth, insertionBonus);
            }

            var bestScore = NegInf;
            int[] best = [];
            foreach (var e in beam.Values)
            {
                var score = Score(e, insertionBonus);
                if (score > bestScore) { bestScore = score; best = e.Labels; }
            }
            return best;
        }

        private sealed class Beam(int[] labels, double logPb, double logPnb)
        {
            public int[] Labels { get; } = labels;
            public double LogPb { get; set; } = logPb;
            public double LogPnb { get; set; } = logPnb;
        }

        private static Beam GetOrAdd(Dictionary<string, Beam> beams, string key, int[] labels)
        {
            if (!beams.TryGetValue(key, out var beam))
            {
                beam = new Beam(labels, NegInf, NegInf);
                beams[key] = beam;
            }
            return beam;
        }

        private static Dictionary<string, Beam> Prune(Dictionary<string, Beam> beams, int width, double insertionBonus)
        {
            if (beams.Count <= width) { return beams; }

            var ordered = new List<KeyValuePair<string, Beam>>(beams.Count);
            foreach (var kv in beams) { ordered.Add(kv); }
            ordered.Sort((a, b) => Score(b.Value, insertionBonus).CompareTo(Score(a.Value, insertionBonus)));

            var result = new Dictionary<string, Beam>(width);
            for (var i = 0; i < width; i++) { result[ordered[i].Key] = ordered[i].Value; }
            return result;
        }

        private static double Score(Beam beam, double insertionBonus)
            => LogSumExp(beam.LogPb, beam.LogPnb) + insertionBonus * beam.Labels.Length;

        private static int[] Append(int[] labels, int label)
        {
            var extended = new int[labels.Length + 1];
            labels.AsSpan().CopyTo(extended);
            extended[labels.Length] = label;
            return extended;
        }

        private static double[] LogSoftmaxPerTimestep(ReadOnlySpan<float> logits, int timeSteps, int classCount)
        {
            var logp = new double[timeSteps * classCount];
            for (var t = 0; t < timeSteps; t++)
            {
                var baseT = t * classCount;
                var max = float.NegativeInfinity;
                for (var k = 0; k < classCount; k++)
                {
                    if (logits[baseT + k] > max) { max = logits[baseT + k]; }
                }
                var sum = 0.0;
                for (var k = 0; k < classCount; k++)
                {
                    sum += Math.Exp(logits[baseT + k] - max);
                }
                var logZ = max + Math.Log(sum);
                for (var k = 0; k < classCount; k++)
                {
                    logp[baseT + k] = logits[baseT + k] - logZ;
                }
            }
            return logp;
        }

        private static double LogSumExp(double a, double b)
        {
            if (double.IsNegativeInfinity(a)) { return b; }
            if (double.IsNegativeInfinity(b)) { return a; }
            var max = a > b ? a : b;
            return max + Math.Log(Math.Exp(a - max) + Math.Exp(b - max));
        }
    }
}
