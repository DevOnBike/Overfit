// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Reads a threshold written the way a human writes one: <c>100MB</c>, <c>50ms</c>, <c>0.01</c>.
    ///
    /// <para><b>The unit suffix is not decoration.</b> Every absolute gate in this guard is a bare
    /// <c>double</c> in the metric's own units, so the configuration for a memory floor is the number
    /// 100000000 — which is exactly where somebody is off by a factor of ten and no reviewer sees it. A
    /// threshold that says <c>100MB</c> is one a person can check at a glance, and one they can be wrong
    /// about out loud.</para>
    ///
    /// <para>Bytes are decimal, not binary: <c>MB</c> is 10⁶, because that is what every Kubernetes dashboard
    /// and every cAdvisor reading shows, and matching the tool the operator is looking at beats matching the
    /// convention a programmer would prefer.</para>
    /// </summary>
    public static class MetricQuantity
    {
        /// <summary>
        /// Parses a threshold. Returns <c>false</c> for anything unrecognised rather than guessing — a
        /// silently misread threshold is worse than a configuration error, because it runs.
        /// </summary>
        public static bool TryParse(string? text, out double value)
        {
            value = 0.0;

            if (string.IsNullOrWhiteSpace(text))
            {
                return false;
            }

            var span = text.AsSpan().Trim();
            var multiplier = 1.0;
            var suffix = 0;

            // Longest suffix first: "ms" must not be read as "m".
            foreach (var (unit, scale) in Units)
            {
                if (span.EndsWith(unit, StringComparison.OrdinalIgnoreCase))
                {
                    multiplier = scale;
                    suffix = unit.Length;

                    break;
                }
            }

            var number = span[..(span.Length - suffix)].Trim();

            if (!double.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed))
            {
                return false;
            }

            if (!double.IsFinite(parsed))
            {
                return false;
            }

            value = parsed * multiplier;

            return true;
        }

        /// <summary>
        /// Parses, or falls back. Used where an absent or unreadable entry means "gate off" rather than
        /// "configuration is broken".
        /// </summary>
        public static double ParseOrDefault(string? text, double fallback = 0.0)
        {
            return TryParse(text, out var value) ? value : fallback;
        }

        /// <summary>
        /// Ordered longest-first so a longer unit is matched before a shorter one that is its prefix or
        /// suffix — <c>ms</c> before <c>s</c>, <c>GB</c> before <c>B</c>.
        /// </summary>
        private static (string Unit, double Scale)[] Units
        {
            get;
        } =
        [
            ("us", 0.000_001),
            ("ms", 0.001),
            // Binary units first: "GiB" must not be read as "B", and Kubernetes resource limits are written
            // this way ("2Gi", "512Mi"), so an operator configuring a memory floor reaches for them by habit.
            // Rejecting them is not a harmless refusal — the entry is dropped and the gate quietly stops
            // gating, which is exactly the failure the reporting exists to prevent.
            ("GiB", 1_073_741_824.0),
            ("MiB", 1_048_576.0),
            ("KiB", 1_024.0),
            ("Gi", 1_073_741_824.0),
            ("Mi", 1_048_576.0),
            ("Ki", 1_024.0),
            ("GB", 1_000_000_000.0),
            ("MB", 1_000_000.0),
            ("KB", 1_000.0),
            ("kB", 1_000.0),
            ("s", 1.0),
            ("B", 1.0),
            ("%", 0.01),
        ];
    }
}
