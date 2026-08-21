// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// A synthetic telemetry source, so the live view can be exercised on a machine with no NVIDIA card.
    /// <para>
    /// It is not a convenience. The view has defects of its own - layout overflow, a divide by zero
    /// before the first sample, a missing value where a number is expected, a card name longer than its
    /// panel - and every one of them is reachable here while none is reachable if the only source is an
    /// absent driver. This exists because the FP16 device path shipped compile-checked only, and one
    /// unexecuted surface per project is already one too many.
    /// </para>
    /// <para>
    /// So the stub deliberately produces the awkward cases rather than tidy ones: a card name far wider
    /// than any sensible panel, a fan reading that is permanently NOT_SUPPORTED (which is what a laptop
    /// answers), and a first sample whose VRAM total is zero, which is what a driver returns before it
    /// has sized the device.
    /// </para>
    /// </summary>
    internal sealed class StubTelemetry : IGpuTelemetry
    {
        private const long TotalBytes = 8L * 1024 * 1024 * 1024;

        private readonly Random _random;
        private int _samples;

        public StubTelemetry(int seed)
        {
            _random = new Random(seed);
        }

        public string SourceName =>
            "STUB - synthetic readings, NOT a device. Present so the view can be exercised without a card.";

        public TelemetrySample Read()
        {
            _samples++;

            // The first sample carries no total, which is what a driver returns before it has sized the
            // device. It is the input that makes VramUsedFraction divide by zero if nothing guards it.
            var total = _samples == 1
                ? TelemetryReading.Absent("driver has not reported a size yet")
                : TelemetryReading.Of(TotalBytes);

            var used = _samples == 1
                ? TelemetryReading.Absent("driver has not reported a size yet")
                : TelemetryReading.Of(TotalBytes * (0.30 + (0.45 * _random.NextDouble())));

            return new TelemetrySample(
                // The brackets are deliberate too: a card name containing '[' is valid MARKUP to
                // Spectre and throws unless the view escapes it. That defect is reachable only from a
                // name like this one.
                "STUB NVIDIA GeForce RTX 0000 Ti [Founders Edition] (a deliberately over-long name)",
                total,
                used,
                TelemetryReading.Of(38 + (_random.NextDouble() * 44)),
                TelemetryReading.Of(_random.NextDouble() * 100),
                TelemetryReading.Absent("NOT_SUPPORTED"));
        }

        public void Dispose()
        {
        }
    }
}
