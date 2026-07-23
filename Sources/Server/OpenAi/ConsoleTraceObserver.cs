// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>
    /// The opt-in per-request phase trace (<c>OVERFIT_SERVER_TRACE=1</c>) as an
    /// <see cref="IChatExchangeObserver"/>: prints replay time, server-side time-to-first-token, and how many
    /// prompt tokens the KV cache reused. Split out of the CLI server so the same trace works behind any host
    /// that drives <see cref="ChatCompletionExchange"/>.
    ///
    /// <para>Timings arrive already elapsed, so this never reads a clock; a single shared instance is safe
    /// because the server decodes one request at a time.</para>
    /// </summary>
    public sealed class ConsoleTraceObserver : IChatExchangeObserver
    {
        public static readonly ConsoleTraceObserver Instance = new();

        public void OnHistoryReplayed(int messageCount, double elapsedMs)
        {
            Console.WriteLine($"[trace] replay {elapsedMs:F1} ms ({messageCount} message(s))");
        }

        public void OnFirstToken(double elapsedMs)
        {
            Console.WriteLine($"[trace] first token {elapsedMs:F1} ms");
        }

        public void OnCompleted(bool streamed, GenerationStats stats, int cachedPromptTokens)
        {
            Console.WriteLine($"[trace] prompt {stats.PromptTokens} tok, {cachedPromptTokens} reused from the KV cache");
        }
    }
}
