// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;

namespace DevOnBike.Overfit.LanguageModels.Agents
{
    /// <summary>
    /// Self-reflection / critic loop: <em>generate → critic evaluates → revise</em> until approved or
    /// caps hit. Model-agnostic — both the generator and the critic are user-supplied delegates, so
    /// you can wire two different models, the same model with two system prompts, or even a
    /// rule-based critic (regex, JSON schema, unit-test runner) for deterministic acceptance.
    ///
    /// On rejection the loop builds the next generator input as
    /// <c>"&lt;original&gt;\nPrevious attempt:\n&lt;candidate&gt;\nFeedback: &lt;feedback&gt;\nPlease revise."</c>
    /// — the generator sees both its own previous output and the critic's notes, the standard
    /// reflexion / self-refine prompt shape.
    /// </summary>
    public sealed class CriticLoop
    {
        private readonly Func<string, string> _generate;
        private readonly Func<string, CriticVerdict> _critique;
        private readonly int _maxIterations;
        private readonly TimeSpan? _timeout;

        public CriticLoop(
            Func<string, string> generate,
            Func<string, CriticVerdict> critique,
            int maxIterations = 4,
            TimeSpan? timeout = null)
        {
            ArgumentNullException.ThrowIfNull(generate);
            ArgumentNullException.ThrowIfNull(critique);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxIterations);

            _generate = generate;
            _critique = critique;
            _maxIterations = maxIterations;
            _timeout = timeout;
        }

        public int MaxIterations => _maxIterations;

        /// <summary>Runs the loop on <paramref name="initialInput"/> and returns the final candidate + trace.</summary>
        public CriticResult Run(string initialInput)
        {
            ArgumentNullException.ThrowIfNull(initialInput);

            var trace = new List<CriticIteration>();
            var currentInput = initialInput;

            var breaker = CircuitBreaker.Run(
                _maxIterations,
                _timeout,
                iterate: _ =>
                {
                    var candidate = _generate(currentInput);
                    var verdict = _critique(candidate);
                    trace.Add(new CriticIteration(candidate, verdict));

                    if (!verdict.Approved)
                    {
                        currentInput = BuildRevisionInput(initialInput, candidate, verdict.Feedback);
                    }

                    return (candidate, verdict);
                },
                isAccepted: pair => pair.verdict.Approved);

            return new CriticResult(breaker.LastValue.candidate, trace, breaker.Outcome);
        }

        private static string BuildRevisionInput(string original, string previousCandidate, string feedback)
        {
            return original
                + "\n\nPrevious attempt:\n" + previousCandidate
                + "\n\nFeedback: " + feedback
                + "\n\nPlease revise.";
        }
    }
}
