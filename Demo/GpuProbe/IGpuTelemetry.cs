// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// A source of live device readings.
    /// <para>
    /// <b><see cref="Read"/> costs a driver round trip and must NEVER be called inside a timed region.</b>
    /// Not inside <see cref="Arm.TimeOnce"/>, not between the timestamp and the synchronise, not inside a
    /// warm-up round's timed call. The probe exists to measure those regions; a driver call inside one
    /// would poison every number it produces and would look exactly like a real result.
    /// </para>
    /// <para>
    /// The interface exists so the view can be driven by <see cref="StubTelemetry"/> on a machine with no
    /// NVIDIA card. A view whose only source is an absent driver is a view nobody has ever seen render.
    /// </para>
    /// </summary>
    internal interface IGpuTelemetry : IDisposable
    {
        /// <summary>What is behind this source, for the report.</summary>
        string SourceName { get; }

        /// <summary>Reads every sensor once. Never throws; an unreadable sensor comes back absent.</summary>
        TelemetrySample Read();
    }
}
