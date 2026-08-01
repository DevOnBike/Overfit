// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// A sample of a stream that answers order statistics without keeping the stream.
    ///
    /// <para><b>Written because the alternative leaks.</b> <c>FloorCalibrator</c> appended every observation
    /// to a list: one per pod per metric per cycle, for as long as the process runs. A shadow week at a
    /// five-minute cadence on twelve replicas is about a million values, and on a hundred replicas it is eight
    /// million and still climbing — a monitoring tool that grows without bound is a poor advertisement for
    /// itself.</para>
    ///
    /// <para><b>The maximum is exact; the quantiles are not.</b> That split is deliberate and matches what the
    /// caller needs: a floor is set from the largest thing a healthy period did, so the maximum has to be
    /// right, while the percentiles are context for a human and tolerate a sampled estimate. Keeping the
    /// maximum separately also means decimation can never throw away the one value the decision depends on.</para>
    ///
    /// <para><b>Decimation is systematic, not random.</b> Once the reservoir is full it is halved — every
    /// second value survives — and the acceptance stride doubles, so the sample stays spread across the whole
    /// stream rather than over-representing its beginning or its end. Deterministic on purpose: a calibration
    /// that produces a different answer on a replay is not one anybody can argue with.</para>
    /// </summary>
    public sealed class BoundedSamples
    {
        private readonly int _capacity;
        private double[] _values;
        private int _stored;
        private int _stride = 1;
        private int _sinceAccepted;

        /// <param name="capacity">Values retained. 1024 keeps a percentile honest to about a tenth of one.</param>
        public BoundedSamples(int capacity = 1024)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 4);

            _capacity = capacity;
            _values = new double[capacity];
        }

        /// <summary>Everything ever offered, not what is retained.</summary>
        public int Count
        {
            get;
            private set;
        }

        /// <summary>The largest value seen. Exact, whatever decimation has done to the reservoir.</summary>
        public double Max
        {
            get;
            private set;
        } = double.NegativeInfinity;

        /// <summary>Offers a value. Non-finite values are ignored — they are absence, not magnitude.</summary>
        public void Add(double value)
        {
            if (!double.IsFinite(value))
            {
                return;
            }

            Count++;

            if (value > Max)
            {
                Max = value;
            }

            if (++_sinceAccepted < _stride)
            {
                return;
            }

            _sinceAccepted = 0;

            if (_stored == _capacity)
            {
                Halve();
            }

            _values[_stored++] = value;
        }

        /// <summary>
        /// Nearest-rank quantile of the retained sample, or zero when nothing has been retained.
        /// </summary>
        public double Quantile(double q)
        {
            if (_stored == 0)
            {
                return 0.0;
            }

            var sorted = new double[_stored];

            Array.Copy(_values, sorted, _stored);
            Array.Sort(sorted);

            var index = (int)(q * (_stored - 1));

            return sorted[Math.Clamp(index, 0, _stored - 1)];
        }

        /// <summary>Serialises the retained sample and the exact counters around it.</summary>
        public string Write()
        {
            var text = new StringBuilder();

            text.Append(Count.ToString(CultureInfo.InvariantCulture)).Append(' ')
                .Append(_stride.ToString(CultureInfo.InvariantCulture)).Append(' ')
                .Append(Max.ToString("R", CultureInfo.InvariantCulture));

            for (var i = 0; i < _stored; i++)
            {
                text.Append(' ').Append(_values[i].ToString("R", CultureInfo.InvariantCulture));
            }

            return text.ToString();
        }

        /// <summary>
        /// Restores a sample. Anything unparseable yields an empty one — a calibration that refuses to start
        /// because its own scratch file is malformed has turned a soft degradation into an outage.
        /// </summary>
        public static BoundedSamples Read(string? text, int capacity = 1024)
        {
            var samples = new BoundedSamples(capacity);

            if (string.IsNullOrWhiteSpace(text))
            {
                return samples;
            }

            var parts = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            if (parts.Length < 3
                || !int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var count)
                || !int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var stride)
                || !double.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var max))
            {
                return samples;
            }

            samples.Count = count;
            samples._stride = Math.Max(1, stride);
            samples.Max = max;

            for (var i = 3; i < parts.Length && samples._stored < capacity; i++)
            {
                if (double.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
                {
                    samples._values[samples._stored++] = value;
                }
            }

            return samples;
        }

        /// <summary>Keeps every second retained value and halves how often new ones are accepted.</summary>
        private void Halve()
        {
            var kept = 0;

            for (var i = 0; i < _stored; i += 2)
            {
                _values[kept++] = _values[i];
            }

            _stored = kept;
            _stride *= 2;
        }
    }
}
