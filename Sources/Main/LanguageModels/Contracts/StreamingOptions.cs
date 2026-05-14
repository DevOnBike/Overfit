// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    /// <summary>
    /// Configuration for streaming token generation via
    /// <c>CachedLlamaSession.StreamGenerate(...)</c>.
    ///
    /// Use sane defaults via the static factories rather than constructing
    /// directly: <see cref="Default"/>, <see cref="WithStopTokens"/>.
    /// </summary>
    public readonly struct StreamingOptions
    {
        /// <summary>
        /// Maximum number of tokens to generate before stopping.
        /// Stream will end naturally if a stop-token is sampled first.
        /// Default: 256.
        /// </summary>
        public int MaxTokens { get; init; }

        /// <summary>
        /// Token IDs that terminate generation when sampled.
        /// The stop token itself is NOT yielded to the consumer.
        /// Default for Qwen-family: empty (caller should set explicit EOS).
        /// </summary>
        public IReadOnlyList<int> StopTokens { get; init; }

        /// <summary>
        /// Sampling configuration (temperature, top-p, repetition penalty etc.).
        /// </summary>
        public SamplingOptions Sampling { get; init; }

        public StreamingOptions(
            int maxTokens,
            IReadOnlyList<int> stopTokens,
            SamplingOptions sampling)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(maxTokens, 1);
            ArgumentNullException.ThrowIfNull(stopTokens);

            MaxTokens = maxTokens;
            StopTokens = stopTokens;
            Sampling = sampling;
        }

        /// <summary>
        /// Greedy sampling, 256-token cap, no stop tokens (consumer breaks on its own).
        /// </summary>
        public static StreamingOptions Default => new(
            maxTokens: 256,
            stopTokens: Array.Empty<int>(),
            sampling: SamplingOptions.Greedy);

        /// <summary>
        /// Greedy sampling with explicit stop tokens.
        /// Example for Qwen 2.5: <c>StreamingOptions.WithStopTokens(256, 151643, 151644, 151645)</c>.
        /// </summary>
        public static StreamingOptions WithStopTokens(int maxTokens, params int[] stopTokens)
            => new(maxTokens, stopTokens, SamplingOptions.Greedy);
    }
}
