// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// Reads <c>test_fixtures/lab/lab-window.csv</c> — a recorded window of the real cluster — into a
    /// <see cref="MetricWindow"/>.
    ///
    /// <para><b>This is what the recording was for.</b> Every threshold in the guard was calibrated against a
    /// simulator that turned out to be wrong three times in one day. A checked-in window of real Prometheus
    /// data makes the detectors runnable, in an ordinary unit test, against something nobody tuned them
    /// against — and it carries which pod was deliberately degraded, so it is also the project's first
    /// labelled data.</para>
    ///
    /// <para><c>nan</c> in the file means the scrape returned nothing, and it is read back as
    /// <see cref="double.NaN"/> rather than zero. Substituting zero would hand the detectors a calm, flat,
    /// fictional signal — the exact failure the file's own header warns about.</para>
    /// </summary>
    public static class LabWindowFixture
    {
        /// <summary>Label the recorder writes for the replica carrying the injected fault.</summary>
        public const string FaultLabel = "FAULT:cpu-throttle";

        /// <summary>
        /// Which recording to read. Defaults to the original four-replica window; set
        /// <c>OVERFIT_LAB_FIXTURE_NAME</c> to read another recording from the same directory.
        ///
        /// <para>Made selectable when a second recording arrived — sixty minutes of the rebuilt twelve-replica
        /// lab. The two are not interchangeable: the first is four replicas of an inference server with an
        /// injected throttle, the second is twelve replicas of a purpose-built workload with nothing wrong,
        /// and a measurement calibrated against one says nothing about the other.</para>
        /// </summary>
        public static string Path =>
            System.IO.Path.Combine(
                AppContext.BaseDirectory, "test_fixtures", "lab",
                Environment.GetEnvironmentVariable("OVERFIT_LAB_FIXTURE_NAME") ?? "lab-window.csv");

        /// <summary>Whether the fixture is present — it is copied to output, so normally yes.</summary>
        public static bool Exists => File.Exists(Path);

        /// <summary>
        /// Loads the window, and the pods the header marks as faulted.
        /// </summary>
        public static (MetricWindow Window, IReadOnlyList<string> FaultedPods) Load()
        {
            var lines = File.ReadAllLines(Path);

            var faulted = new List<string>();
            var stepSeconds = 15;
            long[] timestamps = [];

            var series = new Dictionary<(string Pod, MetricIndex Metric), double[]>();
            var pods = new SortedSet<string>(StringComparer.Ordinal);

            foreach (var line in lines)
            {
                if (line.StartsWith("#", StringComparison.Ordinal))
                {
                    ReadHeader(line, faulted, ref stepSeconds);

                    continue;
                }

                if (line.Length == 0)
                {
                    continue;
                }

                var parts = line.Split(',');

                if (parts[0] == "timestamps_ms")
                {
                    timestamps = new long[parts.Length - 1];

                    for (var i = 1; i < parts.Length; i++)
                    {
                        timestamps[i - 1] = long.Parse(parts[i], CultureInfo.InvariantCulture);
                    }

                    continue;
                }

                if (!Enum.TryParse<MetricIndex>(parts[0], out var metric))
                {
                    continue;
                }

                var pod = parts[1];
                pods.Add(pod);

                var values = new double[parts.Length - 2];

                for (var i = 2; i < parts.Length; i++)
                {
                    values[i - 2] = parts[i] == "nan"
                        ? double.NaN
                        : double.Parse(parts[i], CultureInfo.InvariantCulture);
                }

                series[(pod, metric)] = values;
            }

            var podList = new List<string>(pods);
            var window = new MetricWindow(
                podList,
                timestamps.Length,
                DateTimeOffset.FromUnixTimeMilliseconds(timestamps.Length > 0 ? timestamps[0] : 0),
                TimeSpan.FromSeconds(stepSeconds));

            for (var p = 0; p < podList.Count; p++)
            {
                for (var m = 0; m < (int)MetricIndex.Count; m++)
                {
                    if (!series.TryGetValue((podList[p], (MetricIndex)m), out var values))
                    {
                        continue;
                    }

                    var destination = window.Series(p, (MetricIndex)m);
                    var count = Math.Min(values.Length, destination.Length);

                    values.AsSpan(0, count).CopyTo(destination[..count]);
                }
            }

            return (window, faulted);
        }

        private static void ReadHeader(string line, List<string> faulted, ref int stepSeconds)
        {
            if (line.Contains("label=" + FaultLabel, StringComparison.Ordinal))
            {
                var start = line.IndexOf("pod=", StringComparison.Ordinal);

                if (start >= 0)
                {
                    var rest = line[(start + 4)..];
                    var end = rest.IndexOf(' ');

                    faulted.Add(end > 0 ? rest[..end] : rest);
                }
            }

            var stepAt = line.IndexOf("step_seconds=", StringComparison.Ordinal);

            if (stepAt >= 0)
            {
                var rest = line[(stepAt + 13)..];
                var end = rest.IndexOf(' ');
                var text = end > 0 ? rest[..end] : rest;

                if (int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
                {
                    stepSeconds = parsed;
                }
            }
        }
    }
}
