// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Picks the accelerator and says out loud which one it picked.
    /// <para>
    /// This exists because of a measured trap. ILGPU registers a CPU device in EVERY context, including
    /// one built with <c>Context.Create(b =&gt; b.Cuda())</c>: on a box with no NVIDIA card that call
    /// does not throw, it returns a context holding one device called <c>CPUAccelerator</c>. A probe that
    /// took the first device and labelled its timings "GPU" would print a speedup measured by a CPU
    /// emulator, and nothing in the output would say so.
    /// </para>
    /// </summary>
    internal sealed class DeviceSelection : IDisposable
    {
        private DeviceSelection(Context context, Accelerator accelerator, string note)
        {
            Context = context;
            Accelerator = accelerator;
            Note = note;
        }

        public Context Context { get; }

        public Accelerator Accelerator { get; }

        /// <summary>How the choice was made, in one line, for the report.</summary>
        public string Note { get; }

        /// <summary>False when the "device" is ILGPU's CPU emulator rather than real hardware.</summary>
        public bool IsRealDevice => Accelerator.AcceleratorType != AcceleratorType.CPU;

        public static DeviceSelection Open(ProbeOptions options)
        {
            var context = Context.Create(builder => builder.Cuda().OpenCL().CPU().EnableAlgorithms());

            var devices = context.Devices;
            if (devices.Length == 0)
            {
                throw new InvalidOperationException("ILGPU offered no devices at all, not even a CPU one.");
            }

            if (options.Device.Length > 0)
            {
                var wanted = options.Device switch
                {
                    "cuda" => AcceleratorType.Cuda,
                    "opencl" => AcceleratorType.OpenCL,
                    "cpu" => AcceleratorType.CPU,
                    _ => throw new ArgumentException($"--device must be cuda, opencl or cpu, got '{options.Device}'"),
                };

                var forced = devices.FirstOrDefault(d => d.AcceleratorType == wanted)
                             ?? throw new InvalidOperationException(
                                 $"--device={options.Device} was asked for and ILGPU offers no such device. " +
                                 $"Offered: {string.Join(", ", devices.Select(d => d.AcceleratorType.ToString()))}.");

                return new DeviceSelection(
                    context,
                    forced.CreateAccelerator(context),
                    $"forced by --device={options.Device}");
            }

            var cuda = devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
            if (cuda is not null)
            {
                return new DeviceSelection(context, cuda.CreateAccelerator(context), "CUDA device, chosen automatically");
            }

            var openCl = devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.OpenCL);
            if (openCl is not null)
            {
                return new DeviceSelection(
                    context,
                    openCl.CreateAccelerator(context),
                    "NO CUDA device on this machine; fell back to OpenCL, which is a weaker bound");
            }

            var cpu = devices.First(d => d.AcceleratorType == AcceleratorType.CPU);
            return new DeviceSelection(
                context,
                cpu.CreateAccelerator(context),
                "NO GPU of any kind on this machine; fell back to ILGPU's CPU emulator");
        }

        public void Dispose()
        {
            Accelerator.Dispose();
            Context.Dispose();
        }
    }
}
