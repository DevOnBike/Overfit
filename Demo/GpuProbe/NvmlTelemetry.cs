// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Live readings from a real NVIDIA device, through two different libraries because no single one
    /// has all of it.
    /// <list type="bullet">
    /// <item>Card name and VRAM total come from ILGPU's accelerator - no driver call, no NVML.</item>
    /// <item>VRAM free comes from <c>CudaAccelerator.GetFreeMemory</c>, which is the CUDA driver API
    /// (<c>cuMemGetInfo</c>) and needs only <c>nvcuda.dll</c>, already required by the probe.</item>
    /// <item>Temperature and fan come from NVML, which is <c>nvml.dll</c> from the driver.</item>
    /// </list>
    /// <para>
    /// <b>EVERY SENSOR IS GUARDED SEPARATELY, and that is the point rather than defensiveness.</b> A
    /// consumer card answers NOT_SUPPORTED for sensors it does not have - a laptop has no readable fan,
    /// many cards have no readable one either - and NVML signals that by throwing. One sensor failing
    /// must not blank the other five, and a failure must never become a zero. A zero printed for an
    /// unreadable temperature is a fabricated reading, and in an artefact returned from somebody else's
    /// machine it is indistinguishable from a real one.
    /// </para>
    /// <para>
    /// <b>NOT ONE LINE OF THIS TYPE HAS EVER EXECUTED.</b> The development machine has no NVIDIA device,
    /// so <see cref="TryCreate"/> returns null here. <see cref="StubTelemetry"/> exists precisely so the
    /// view above it is not also unexecuted - the view is the part with the layout defects, and it is
    /// fully exercisable without a card.
    /// </para>
    /// </summary>
    internal sealed class NvmlTelemetry : IGpuTelemetry
    {
        private readonly CudaAccelerator _cuda;
        private readonly NvmlDevice? _nvml;
        private readonly string _nvmlNote;

        private NvmlTelemetry(CudaAccelerator cuda, NvmlDevice? nvml, string nvmlNote)
        {
            _cuda = cuda;
            _nvml = nvml;
            _nvmlNote = nvmlNote;
        }

        /// <summary>Why there is no live telemetry at all, or null when there is.</summary>
        public static string? Unavailable { get; private set; }

        public string SourceName => _nvml is null
            ? $"ILGPU CUDA driver API only - VRAM and name are live, the sensors are not ({_nvmlNote})"
            : "ILGPU CUDA driver API for memory, NVML for the sensors";

        /// <summary>
        /// Returns null and sets <see cref="Unavailable"/> rather than throwing. NVML failing on its own
        /// is NOT fatal: memory and name still come back, so the source degrades to a partial one rather
        /// than to nothing.
        /// </summary>
        public static NvmlTelemetry? TryCreate(Accelerator accelerator)
        {
            if (accelerator is not CudaAccelerator cuda)
            {
                Unavailable =
                    $"the selected accelerator is {accelerator.AcceleratorType}, not CUDA, and there is no " +
                    "vendor-neutral way to read a card's temperature. Pass --live-stub to see the view anyway.";
                return null;
            }

            try
            {
                return new NvmlTelemetry(cuda, NvmlDevice.CreateFromAccelerator(cuda), string.Empty);
            }
            catch (Exception ex) when (ex is NvmlException or DllNotFoundException
                                          or TypeInitializationException or EntryPointNotFoundException
                                          or NotSupportedException)
            {
                // Memory and name do not need NVML, so a partial source beats no source.
                return new NvmlTelemetry(cuda, null, $"nvml.dll: {ex.Message.Trim()}");
            }
        }

        public TelemetrySample Read() => new(
            _cuda.Name,
            ReadVramTotal(),
            ReadVramUsed(),
            ReadTemperature(),
            TelemetryReading.Absent(
                "ILGPU's NVML wrapper does not expose nvmlDeviceGetUtilizationRates; reading it needs P/Invoke"),
            ReadFan());

        private TelemetryReading ReadVramTotal()
        {
            try
            {
                return TelemetryReading.Of(_cuda.MemorySize);
            }
            catch (Exception ex) when (ex is CudaException or NotSupportedException or InvalidOperationException)
            {
                return TelemetryReading.Absent(ex.Message.Trim());
            }
        }

        private TelemetryReading ReadVramUsed()
        {
            try
            {
                var free = _cuda.GetFreeMemory();
                var total = _cuda.MemorySize;
                return free <= total && free >= 0
                    ? TelemetryReading.Of(total - free)
                    : TelemetryReading.Absent("the driver reported free memory larger than the total");
            }
            catch (Exception ex) when (ex is CudaException or NotSupportedException or InvalidOperationException)
            {
                return TelemetryReading.Absent(ex.Message.Trim());
            }
        }

        private TelemetryReading ReadTemperature()
        {
            if (_nvml is null)
            {
                return TelemetryReading.Absent("NVML is not available");
            }

            try
            {
                return TelemetryReading.Of(_nvml.GetGpuTemperature());
            }
            catch (Exception ex) when (ex is NvmlException or NotSupportedException or EntryPointNotFoundException)
            {
                return TelemetryReading.Absent(ex.Message.Trim());
            }
        }

        private TelemetryReading ReadFan()
        {
            if (_nvml is null)
            {
                return TelemetryReading.Absent("NVML is not available");
            }

            try
            {
                return TelemetryReading.Of(_nvml.GetFanSpeed(0));
            }
            catch (Exception ex) when (ex is NvmlException or NotSupportedException or EntryPointNotFoundException)
            {
                return TelemetryReading.Absent(ex.Message.Trim());
            }
        }

        public void Dispose() => _nvml?.Dispose();
    }
}
