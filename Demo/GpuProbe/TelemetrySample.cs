// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>One reading of every sensor the live view shows. Every field may be absent with a reason.</summary>
    internal sealed class TelemetrySample
    {
        public TelemetrySample(
            string cardName,
            TelemetryReading vramTotalBytes,
            TelemetryReading vramUsedBytes,
            TelemetryReading temperatureCelsius,
            TelemetryReading utilisationPercent,
            TelemetryReading fanPercent)
        {
            CardName = cardName;
            VramTotalBytes = vramTotalBytes;
            VramUsedBytes = vramUsedBytes;
            TemperatureCelsius = temperatureCelsius;
            UtilisationPercent = utilisationPercent;
            FanPercent = fanPercent;
        }

        public string CardName { get; }

        public TelemetryReading VramTotalBytes { get; }

        /// <summary>
        /// VRAM in use ON THE WHOLE DEVICE, not by this process. The driver reports the device total and
        /// the label says so, because "how much we occupy" and "how much is occupied" differ by whatever
        /// else is on the card, and a desktop compositor is always something.
        /// </summary>
        public TelemetryReading VramUsedBytes { get; }

        public TelemetryReading TemperatureCelsius { get; }

        public TelemetryReading UtilisationPercent { get; }

        public TelemetryReading FanPercent { get; }

        /// <summary>
        /// Used as a fraction of total, or absent. Guarded against a zero total: the first sample can
        /// arrive before the driver has reported a size, and a divide by zero there would take the view
        /// down mid-run.
        /// </summary>
        public TelemetryReading VramUsedFraction =>
            VramTotalBytes.HasValue && VramUsedBytes.HasValue && VramTotalBytes.Value > 0
                ? TelemetryReading.Of(VramUsedBytes.Value / VramTotalBytes.Value * 100.0)
                : TelemetryReading.Absent("total unknown");
    }
}
