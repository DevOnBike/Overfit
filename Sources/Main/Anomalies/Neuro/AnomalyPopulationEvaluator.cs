// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Anomalies.Neuro
{
    /// <summary>
    /// Bridges the anomaly objective into the evolutionary engine: an <see cref="IPopulationEvaluator"/> that
    /// scores a whole population of detector genomes against one labelled dataset. Drop it into an
    /// <c>EvolutionRunner</c> alongside <c>OpenAiEsStrategy</c> / <c>SeparableCmaEsStrategy</c> and the strategy
    /// optimises F1 / business cost / recall-under-budget <b>directly</b> — no differentiable surrogate, and the
    /// alert threshold evolves with the weights instead of being hand-tuned afterwards.
    ///
    /// <para>The dataset is held by reference and never copied; evaluation allocates nothing per candidate.
    /// Candidates are independent, so this is embarrassingly parallel — wrap or replace with a parallel
    /// evaluator if the sample count makes it worthwhile (measure first: for a few hundred samples the dispatch
    /// can cost more than the work).</para>
    /// </summary>
    public sealed class AnomalyPopulationEvaluator : IPopulationEvaluator
    {
        private readonly float[] _features;
        private readonly byte[] _labels;
        private readonly int _sampleCount;
        private readonly int _inputs;
        private readonly int _hidden;
        private readonly AnomalyFitnessOptions _options;

        /// <param name="features">Flat samples: <paramref name="sampleCount"/> × <paramref name="inputs"/>.</param>
        /// <param name="labels">1 = anomaly, 0 = normal.</param>
        /// <param name="inputs">Features per window.</param>
        /// <param name="hidden">Hidden units in the scorer.</param>
        /// <param name="options">Objective to maximise.</param>
        public AnomalyPopulationEvaluator(
            float[] features,
            byte[] labels,
            int inputs,
            int hidden,
            AnomalyFitnessOptions options = default)
        {
            ArgumentNullException.ThrowIfNull(features);
            ArgumentNullException.ThrowIfNull(labels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputs);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hidden);

            if (features.Length % inputs != 0)
            {
                throw new ArgumentException(
                    $"Feature buffer length {features.Length} is not a multiple of inputs {inputs}.", nameof(features));
            }

            _sampleCount = features.Length / inputs;
            if (labels.Length != _sampleCount)
            {
                throw new ArgumentException(
                    $"Expected {_sampleCount} labels to match the feature buffer, got {labels.Length}.", nameof(labels));
            }

            _features = features;
            _labels = labels;
            _inputs = inputs;
            _hidden = hidden;
            _options = options.Objective == default && options.MissCost == 0f ? new AnomalyFitnessOptions() : options;
        }

        /// <summary>Genome length this evaluator expects — pass it as the strategy's parameter count.</summary>
        public int GenomeSize => AnomalyMlp.GenomeSize(_inputs, _hidden);

        /// <inheritdoc />
        public void Evaluate(
            ReadOnlySpan<float> populationData, Span<float> fitnessOut, int populationSize, int parameterCount)
        {
            if (parameterCount != GenomeSize)
            {
                throw new ArgumentException(
                    $"Genome size mismatch: strategy has {parameterCount} parameters, the detector needs {GenomeSize} "
                    + $"({_inputs} inputs x {_hidden} hidden + biases + threshold).", nameof(parameterCount));
            }

            for (var i = 0; i < populationSize; i++)
            {
                fitnessOut[i] = AnomalyFitness.Evaluate(
                    populationData.Slice(i * parameterCount, parameterCount),
                    _features, _labels, _sampleCount, _inputs, _hidden, in _options);
            }
        }
    }
}
