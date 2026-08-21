// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// One sensor value, or the reason there is no value. There is deliberately no third state and no
    /// default of zero.
    /// <para>
    /// A consumer card does not expose every sensor and NVML answers NOT_SUPPORTED for the ones it does
    /// not. A zero printed for an unreadable temperature is a fabricated reading, and it is
    /// indistinguishable from a real 0 C in the artefact that comes back from somebody else's machine.
    /// This is the same rule that stopped the probe printing a cuBLAS compute type ILGPU cannot report.
    /// </para>
    /// </summary>
    internal readonly struct TelemetryReading
    {
        private readonly double _value;
        private readonly string? _reason;

        private TelemetryReading(double value, string? reason)
        {
            _value = value;
            _reason = reason;
        }

        public static TelemetryReading Of(double value) => new(value, null);

        public static TelemetryReading Absent(string reason) => new(0, reason);

        public bool HasValue => _reason is null;

        /// <summary>Only meaningful when <see cref="HasValue"/> is true.</summary>
        public double Value => _value;

        /// <summary>Only meaningful when <see cref="HasValue"/> is false.</summary>
        public string Reason => _reason ?? string.Empty;

        /// <summary>The value with its unit, or the reason it is absent. Never a bare number.</summary>
        public string Format(string unit, string format = "F0") => HasValue
            ? string.Create(CultureInfo.InvariantCulture, $"{_value.ToString(format, CultureInfo.InvariantCulture)} {unit}")
            : _reason!;
    }
}
