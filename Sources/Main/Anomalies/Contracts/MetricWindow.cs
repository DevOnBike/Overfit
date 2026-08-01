// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// One evaluation window: every pod, every metric, aligned to a shared timestamp grid.
    ///
    /// <para><b>The detectors' input, deliberately separated from where it came from.</b> Prometheus, a
    /// recorded fixture and a synthetic population all produce this shape, so the guard can be exercised
    /// against a file in a unit test and against a cluster in production without a second code path — and the
    /// lab recording under <c>Tests/test_fixtures/lab/</c> is literally this, serialised.</para>
    ///
    /// <para><b>Missing is <see cref="double.NaN"/>, never zero</b>, throughout. A scrape that returned
    /// nothing and a metric that genuinely read zero are different facts, and merging them is how a
    /// misconfigured query becomes a calm, flat, entirely fictional signal that every detector accepts.</para>
    ///
    /// <para>Values are stored in one flat array indexed <c>(pod * MetricCount + metric) * Length</c>. Flat
    /// rather than jagged because jagged <c>double[][]</c> is a build error in this project: one allocation
    /// and one cache-friendly block beats one per row, and rows are handed out as spans anyway.</para>
    /// </summary>
    public sealed class MetricWindow
    {
        private readonly double[] _values;
        private readonly string[] _pods;
        private readonly string[] _custom;
        private readonly Dictionary<string, int> _customIndex;

        /// <param name="pods">Pod names, in a stable order. Index into this is the pod index throughout.</param>
        /// <param name="length">Samples per series.</param>
        /// <param name="start">Wall-clock time of sample 0.</param>
        /// <param name="step">Spacing between samples; should match the scrape interval.</param>
        /// <param name="custom">
        /// Channels outside the modelled set, by reported name. They are stored alongside the known ones and
        /// read by the rules, peer and trend families; the learned family never sees them, because
        /// <c>MetricSnapshot.FeatureCount</c> is a trained model's input contract.
        /// </param>
        public MetricWindow(
            IReadOnlyList<string> pods,
            int length,
            DateTimeOffset start,
            TimeSpan step,
            IReadOnlyList<string>? custom = null)
        {
            ArgumentNullException.ThrowIfNull(pods);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(step, TimeSpan.Zero);

            _pods = new string[pods.Count];

            for (var i = 0; i < pods.Count; i++)
            {
                _pods[i] = pods[i] ?? throw new ArgumentException("Pod names must not be null.", nameof(pods));
            }

            Length = length;
            Start = start;
            Step = step;

            _custom = new string[custom?.Count ?? 0];
            _customIndex = new Dictionary<string, int>(_custom.Length, StringComparer.Ordinal);

            for (var i = 0; i < _custom.Length; i++)
            {
                var name = custom![i];

                if (string.IsNullOrWhiteSpace(name))
                {
                    throw new ArgumentException("Custom channel names must not be blank.", nameof(custom));
                }

                _custom[i] = name;
                _customIndex[name] = i;
            }

            // Sized in long arithmetic and checked before narrowing: pods × channels × samples is exactly the
            // shape of product that overflows a 32-bit multiply into a positive, undersized allocation.
            var channels = (int)MetricIndex.Count + _custom.Length;
            var cells = (long)pods.Count * channels * length;

            if (cells > int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(length),
                    $"{pods.Count} pods × {channels} channels × {length} samples is {cells} values, past "
                    + "the largest array this can hold.");
            }

            _values = new double[(int)cells];
            _values.AsSpan().Fill(double.NaN);
        }

        /// <summary>Pods in the window, in index order.</summary>
        public IReadOnlyList<string> Pods => _pods;

        /// <summary>Samples per series.</summary>
        public int Length
        {
            get;
        }

        /// <summary>Wall-clock time of sample 0.</summary>
        public DateTimeOffset Start
        {
            get;
        }

        /// <summary>Spacing between samples.</summary>
        public TimeSpan Step
        {
            get;
        }

        /// <summary>End of the window — the time of the last sample.</summary>
        public DateTimeOffset End => Start + (Step * (Length - 1));

        /// <summary>Channels outside the modelled set carried by this window, in index order.</summary>
        public IReadOnlyList<string> CustomChannels => _custom;

        /// <summary>One pod's series for one metric. Writable, so a source can fill it in place.</summary>
        public Span<double> Series(int pod, MetricIndex metric)
        {
            var index = (int)metric;

            if ((uint)index >= (uint)MetricIndex.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(metric), metric, "Unknown metric.");
            }

            return Channel(pod, index);
        }

        /// <summary>One pod's series for a custom channel, by the name it is reported under.</summary>
        public Span<double> Series(int pod, string custom)
        {
            ArgumentNullException.ThrowIfNull(custom);

            if (!_customIndex.TryGetValue(custom, out var index))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(custom), custom, "This window carries no such custom channel.");
            }

            return Channel(pod, (int)MetricIndex.Count + index);
        }

        /// <summary>
        /// A signal as memory over this window's own storage — no copy.
        ///
        /// <para><b>For readers that outlive the call but not the window.</b> A <see cref="Span{T}"/> cannot
        /// be stored, so every consumer that needed to keep a series was copying it, and the guard was making
        /// two copies per pod per signal per cycle whether or not anything came of them — measured at
        /// <b>3 MB a cycle on two hundred replicas</b>, for series almost all of which are read once and
        /// discarded.</para>
        ///
        /// <para><b>The lifetime is the window's, and that is the whole caution.</b> Anything that keeps this
        /// beyond the cycle keeps the entire window alive with it — 1.7 MB at two hundred replicas — and will
        /// read whatever the window holds later. Retain it only for as long as the window is retained; copy
        /// when it has to outlive one, which is what a finding does.</para>
        /// </summary>
        public ReadOnlyMemory<double> SeriesMemory(int pod, MetricIndex metric)
        {
            var index = (int)metric;

            if ((uint)index >= (uint)MetricIndex.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(metric), metric, "Unknown metric.");
            }

            return new ReadOnlyMemory<double>(_values, Offset(pod, index), Length);
        }

        /// <summary>A custom channel as memory. Same lifetime caution as the indexed overload.</summary>
        public ReadOnlyMemory<double> SeriesMemory(int pod, string custom)
        {
            ArgumentNullException.ThrowIfNull(custom);

            if (!_customIndex.TryGetValue(custom, out var index))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(custom), custom, "This window carries no such custom channel.");
            }

            return new ReadOnlyMemory<double>(
                _values, Offset(pod, (int)MetricIndex.Count + index), Length);
        }

        private Span<double> Channel(int pod, int channel)
            => _values.AsSpan(Offset(pod, channel), Length);

        /// <summary>Start of one pod's channel inside the flat store. Shared so span and memory cannot drift.</summary>
        private int Offset(int pod, int channel)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(pod);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(pod, _pods.Length);

            var channels = (int)MetricIndex.Count + _custom.Length;

            return (int)((((long)pod * channels) + channel) * Length);
        }

        /// <summary>
        /// Timestamps in seconds from the window start, which is what <see cref="Statistics.TrendDetector"/>
        /// takes. Written into a caller-owned span so nothing is allocated per cycle.
        /// </summary>
        public void WriteTimestampSeconds(Span<double> destination)
        {
            if (destination.Length < Length)
            {
                throw new ArgumentException(
                    $"Destination holds {destination.Length} of the {Length} timestamps needed.",
                    nameof(destination));
            }

            var step = Step.TotalSeconds;

            for (var i = 0; i < Length; i++)
            {
                destination[i] = i * step;
            }
        }

        /// <summary>
        /// How many pods reported at least one finite sample for <paramref name="metric"/>.
        ///
        /// <para><b>This is the coverage question, and it is per pod rather than "did anything arrive".</b> A
        /// metric returning for one pod out of twenty passes an any-series check and is still a blind spot on
        /// nineteen of them.</para>
        /// </summary>
        public int PodsReporting(MetricIndex metric)
        {
            return PodsReportingChannel((int)metric);
        }

        /// <summary>The same question for a custom channel.</summary>
        public int PodsReporting(string custom)
        {
            ArgumentNullException.ThrowIfNull(custom);

            return _customIndex.TryGetValue(custom, out var index)
                ? PodsReportingChannel((int)MetricIndex.Count + index)
                : 0;
        }

        private int PodsReportingChannel(int channel)
        {
            var reporting = 0;

            for (var pod = 0; pod < _pods.Length; pod++)
            {
                var series = Channel(pod, channel);

                for (var i = 0; i < series.Length; i++)
                {
                    if (double.IsFinite(series[i]))
                    {
                        reporting++;

                        break;
                    }
                }
            }

            return reporting;
        }
    }
}
