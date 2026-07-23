// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>
    /// Optional per-exchange hooks so each host can attach what only it cares about — the CLI server prints a
    /// phase trace, the ASP.NET host records Prometheus metrics — without either concern leaking into the
    /// shared <see cref="ChatCompletionExchange"/>. All methods are no-ops by default; pass <c>null</c> for
    /// none. Timings are handed in already elapsed (milliseconds) so the observer never touches a clock.
    /// </summary>
    public interface IChatExchangeObserver
    {
        /// <summary>History replay finished — <paramref name="messageCount"/> turns re-applied to the session.</summary>
        void OnHistoryReplayed(int messageCount, double elapsedMs)
        {
        }

        /// <summary>The first streamed token was produced (streaming path only), <paramref name="elapsedMs"/>
        /// after generation began — the server-side component of time-to-first-token.</summary>
        void OnFirstToken(double elapsedMs)
        {
        }

        /// <summary>
        /// The exchange completed. <paramref name="streamed"/> distinguishes the SSE path from the one-shot
        /// JSON path; <paramref name="cachedPromptTokens"/> is how many prompt tokens the prompt cache reused
        /// instead of re-encoding (0 when nothing matched).
        /// </summary>
        void OnCompleted(bool streamed, GenerationStats stats, int cachedPromptTokens)
        {
        }
    }
}
