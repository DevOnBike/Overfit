// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    ///     Sanitizes an AlignResult in place.
    ///     Level 1: removes invalid pods.
    ///     Level 2: corrects individual metric values in remaining windows.
    ///     Single instance should be reused across scrapes — _podFirstSeen persists between calls.
    /// </summary>
    public sealed class WindowSanitizer
    {
        private readonly SanitizerOptions _options;
        private readonly Dictionary<string, long> _podFirstSeen = [];

        public WindowSanitizer(SanitizerOptions options)
        {
            _options = options;
        }

        // ---------------------------------------------------------------------------
        // Entry point
        // ---------------------------------------------------------------------------

        public void Sanitize(AlignResult result, long scrapeTimestampMs)
        {
            var warmupMs = (long)_options.WarmupDuration.TotalMilliseconds;

            FilterByUptime(result, _podFirstSeen, scrapeTimestampMs, warmupMs);
            FilterByNanRatio(result, _options.MaxNanRatio);

            CorrectValues(result);
        }

        // ---------------------------------------------------------------------------
        // Level 1 — pod filters
        // ---------------------------------------------------------------------------

        private static void FilterByUptime(
            AlignResult result,
            Dictionary<string, long> podFirstSeen,
            long scrapeTimestampMs,
            long warmupMs)
        {
            for (var i = result.Windows.Count - 1; i >= 0; i--)
            {
                var podName = result.PodIndex[i].PodName;

                if (!podFirstSeen.TryGetValue(podName, out var first))
                {
                    podFirstSeen[podName] = scrapeTimestampMs;
                    Remove(result, i);
                    continue;
                }

                if (scrapeTimestampMs - first < warmupMs)
                {
                    Remove(result, i);
                }
            }
        }

        private static void FilterByNanRatio(AlignResult result, float maxNanRatio)
        {
            for (var i = result.Windows.Count - 1; i >= 0; i--)
            {
                var data = result.Windows[i].Data;
                if (data.Length == 0)
                {
                    Remove(result, i);
                    continue;
                }

                var nanCount = 0;
                foreach (var v in data)
                {
                    if (float.IsNaN(v)) nanCount++;
                }

                if ((float)nanCount / data.Length > maxNanRatio)
                    Remove(result, i);
            }
        }

        // ---------------------------------------------------------------------------
        // Level 2 — value correction
        // ---------------------------------------------------------------------------

        private static void CorrectValues(AlignResult result)
        {
            foreach (var window in result.Windows)
            {
                var data = window.Data;

                for (var i = 0; i < data.Length; i++)
                {
                    if (float.IsNaN(data[i]) || data[i] < 0f)
                    {
                        data[i] = 0f;
                    }
                }
            }
        }

        // ---------------------------------------------------------------------------

        public void CleanupStaleEntries(IEnumerable<string> activePodNames)
        {
            var active = new HashSet<string>(activePodNames);
            foreach (var key in _podFirstSeen.Keys.ToList())
            {
                if (!active.Contains(key))
                    _podFirstSeen.Remove(key);
            }
        }

        private static void Remove(AlignResult result, int index)
        {
            result.Windows.RemoveAt(index);
            result.PodIndex.RemoveAt(index);
        }
    }
}