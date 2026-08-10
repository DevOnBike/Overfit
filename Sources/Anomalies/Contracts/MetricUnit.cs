// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// What a channel's numbers actually are, once the query has run.
    ///
    /// <para><b>This exists because a binding can be in the wrong unit for months and nothing notices.</b>
    /// <c>ErrorRate</c> shipped bound to a plain error counter with <c>kind: Counter</c>, which renders as
    /// <c>rate(...)</c> — errors per second, unbounded — while <c>MetricSnapshot</c> documents feature [8] as
    /// <c>rate(5xx) / rate(total)</c>, a fraction on [0,1]. The query was valid, the series existed, the
    /// channel reported numbers every cycle, and a parity test even copied the binding between two files. A
    /// threshold reasoned about as a fraction means nothing against a per-second rate.</para>
    ///
    /// <para><b>The unit is a property of the CHANNEL, not of the series.</b> Which series a deployment binds
    /// is a local decision; what the channel promises its consumers is not, because thresholds, floors and
    /// the learned model's feature vector are all written against the promise. That is why this lives beside
    /// <see cref="MetricIndex"/> rather than in a config file.</para>
    /// </summary>
    public enum MetricUnit : byte
    {
        /// <summary>A share of one, on [0,1]. A threshold like 0.25 is meaningful.</summary>
        Fraction = 0,

        /// <summary>CPU seconds per second — "cores". Not a fraction: it is unbounded above by core count.</summary>
        Cores = 1,

        /// <summary>Bytes. Absolute, and comparable between pods only when they run the same image.</summary>
        Bytes = 2,

        /// <summary>Events per second, from a <c>rate()</c> over a cumulative counter.</summary>
        PerSecond = 3,

        /// <summary>Events within the window, from an <c>increase()</c>. Not a rate — the window's own count.</summary>
        EventsInWindow = 4,

        /// <summary>Milliseconds, from a histogram quantile scaled from seconds.</summary>
        Milliseconds = 5,

        /// <summary>A plain count with no time dimension — a queue depth, a number of items.</summary>
        Count = 6,
    }
}
