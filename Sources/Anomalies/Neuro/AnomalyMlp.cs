// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Neuro
{
    /// <summary>
    /// A tiny anomaly scorer whose entire state — weights, biases AND the alert threshold — lives in one flat
    /// genome, so an evolutionary strategy can optimise all of it at once.
    ///
    /// <para><b>Layout</b> (<see cref="GenomeSize"/> floats): <c>W1[hidden×inputs] · b1[hidden] · W2[hidden] ·
    /// threshold[1]</c>. Putting the <b>threshold in the genome</b> is the point: in a backprop-trained detector
    /// the decision threshold is a separate manual knob tuned after training, so the network is optimised for one
    /// objective and then judged by another. Here the search optimises the network and its operating point
    /// <i>jointly</i>, against the metric you actually care about.</para>
    ///
    /// <para><b>There is deliberately no output bias.</b> It would be redundant with the threshold —
    /// <c>sigmoid(z) ≥ t</c> ⟺ <c>z ≥ logit(t)</c>, and an output bias just shifts <c>z</c> — so the pair would
    /// carry one degree of freedom too many, letting the search satisfy any decision boundary by moving either
    /// gene. Dropping it costs no expressiveness (both forms describe the same family of decision rules) and
    /// keeps the evolved operating point readable as a probability.</para>
    ///
    /// <para><b>Why no autograd.</b> <see cref="Score"/> reads the genome directly as a
    /// <see cref="ReadOnlySpan{T}"/> and allocates nothing. Evolution is gradient-free and calls this
    /// population × samples × generations times, so building an autograd tape per forward pass would be pure
    /// waste — this is deliberately NOT the training path. (To evolve an existing <c>IModule</c> instead, use
    /// <c>NeuralNetworkParameterAdapter</c>.)</para>
    /// </summary>
    public static class AnomalyMlp
    {
        /// <summary>Genome length for the given shape: W1 + b1 + W2 + threshold (no output bias — see class docs).</summary>
        public static int GenomeSize(int inputs, int hidden) => (hidden * inputs) + hidden + hidden + 1;

        /// <summary>The evolved alert threshold — the last gene. Scores at or above it are anomalies.</summary>
        public static float Threshold(ReadOnlySpan<float> genome) => genome[genome.Length - 1];

        /// <summary>
        /// Anomaly score in [0,1] for one feature window. tanh hidden layer, logistic output — both bounded, so a
        /// wild genome early in the search produces a usable score instead of NaN.
        /// </summary>
        /// <param name="genome">Parameters laid out per the class docs.</param>
        /// <param name="features">One window of metrics, length <paramref name="inputs"/>.</param>
        /// <param name="inputs">Feature count.</param>
        /// <param name="hidden">Hidden unit count.</param>
        public static float Score(ReadOnlySpan<float> genome, ReadOnlySpan<float> features, int inputs, int hidden)
        {
            var w1 = genome.Slice(0, hidden * inputs);
            var b1 = genome.Slice(hidden * inputs, hidden);
            var w2 = genome.Slice((hidden * inputs) + hidden, hidden);

            var sum = 0f;
            for (var h = 0; h < hidden; h++)
            {
                var acc = b1[h];
                var row = w1.Slice(h * inputs, inputs);
                for (var i = 0; i < inputs; i++)
                {
                    acc += row[i] * features[i];
                }
                sum += w2[h] * MathF.Tanh(acc);
            }

            return 1f / (1f + MathF.Exp(-sum));
        }

        /// <summary>Convenience: does this genome flag the window as an anomaly (score ≥ evolved threshold)?</summary>
        public static bool IsAnomaly(ReadOnlySpan<float> genome, ReadOnlySpan<float> features, int inputs, int hidden)
            => Score(genome, features, inputs, hidden) >= Threshold(genome);
    }
}
