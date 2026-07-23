// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.Server.OpenAi;

namespace DevOnBike.Overfit.Server.AspNet.Services
{
    /// <summary>
    /// Fans one chat exchange's observer callbacks out to several observers — used to run metric recording and
    /// the opt-in phase trace together, since the exchange takes a single observer.
    /// </summary>
    internal sealed class CompositeChatObserver : IChatExchangeObserver
    {
        private readonly IChatExchangeObserver[] _observers;

        public CompositeChatObserver(params IChatExchangeObserver[] observers) => _observers = observers;

        public void OnHistoryReplayed(int messageCount, double elapsedMs)
        {
            foreach (var observer in _observers)
            {
                observer.OnHistoryReplayed(messageCount, elapsedMs);
            }
        }

        public void OnFirstToken(double elapsedMs)
        {
            foreach (var observer in _observers)
            {
                observer.OnFirstToken(elapsedMs);
            }
        }

        public void OnCompleted(bool streamed, GenerationStats stats, int cachedPromptTokens)
        {
            foreach (var observer in _observers)
            {
                observer.OnCompleted(streamed, stats, cachedPromptTokens);
            }
        }
    }
}
