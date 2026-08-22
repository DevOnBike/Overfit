// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text;
using DevOnBike.Overfit.Autograd;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The identity of the machine and the configuration that produced the numbers. A measurement
    /// without it is not attributable to anything.
    /// <para>
    /// Deliberately NOT collected, because this report travels back from somebody else's computer: host
    /// name, user name, and any filesystem path. The resolved assembly locations are reduced to "from the
    /// NuGet cache" or "beside the executable", which is the part that answers Risk R5, without carrying
    /// the directory the answer was read from.
    /// </para>
    /// </summary>
    internal sealed class MachineEcho
    {
        private MachineEcho(IReadOnlyList<KeyValuePair<string, string>> fields)
        {
            Fields = fields;
        }

        public IReadOnlyList<KeyValuePair<string, string>> Fields { get; }

        public static MachineEcho Collect(Accelerator accelerator, Context context, ProbeOptions options)
        {
            var f = new List<KeyValuePair<string, string>>();

            void Add(string key, string value) => f.Add(new KeyValuePair<string, string>(key, value));

            Add("utc", DateTime.UtcNow.ToString("u"));
            Add("probe version", "gpu-probe 1, plan docs/specs/gpu-probe-route-and-design-plan.md");

            Add("accelerator type", accelerator.AcceleratorType.ToString());
            Add("accelerator name", accelerator.Name);
            Add("accelerator memory bytes", accelerator.MemorySize.ToString());
            Add("accelerator max group size", accelerator.MaxNumThreadsPerGroup.ToString());
            Add("accelerator warp size", accelerator.WarpSize.ToString());
            Add("accelerator max shared memory per group", accelerator.MaxSharedMemoryPerGroup.ToString());

            // The compute-unit count for EVERY accelerator type, not only CUDA. It used to live in the
            // CUDA branch below, so a run on an OpenCL or CPU device reported no unit count at all - and
            // "how many of my GPU's units are being used" is the question this probe was asked twice.
            AddComputeUnits(Add, accelerator);

            if (accelerator is CudaAccelerator cuda)
            {
                var device = (CudaDevice)cuda.Device;
                Add("cuda compute capability", $"{device.Architecture}");
                Add("cuda driver version", device.DriverVersion.ToString());
                Add("cuda clock rate kHz", device.ClockRate.ToString());
                Add("cuda multiprocessors", device.NumMultiprocessors.ToString());
            }

            Add("devices ILGPU offered", DescribeDevices(context));

            Add("cpu model", CpuModel());
            Add("cpu logical processors", Environment.ProcessorCount.ToString());
            Add("os", RuntimeInformation.OSDescription);
            Add("os architecture", RuntimeInformation.OSArchitecture.ToString());
            Add("process architecture", RuntimeInformation.ProcessArchitecture.ToString());
            Add("dotnet", RuntimeInformation.FrameworkDescription);
            Add("server gc", System.Runtime.GCSettings.IsServerGC.ToString());

            // Risk R5: the CPU arm runs the PUBLISHED package, not this repository's HEAD. Which build it
            // actually resolved is therefore part of the result, not a footnote.
            Add("DevOnBike.Overfit", DescribeAssembly(typeof(ComputationGraph)));
            Add("ILGPU", DescribeAssembly(typeof(Context)));

            Add("options", options.Describe());

            // Section 3.7: an empty field a human is asked to fill is honest; a missing field reads as
            // "not relevant".
            Add("RAM type and speed", "not readable without WMI or a native call - PLEASE FILL IN MANUALLY");
            Add("other load on the machine during the run", "PLEASE FILL IN MANUALLY");

            return new MachineEcho(f);
        }

        /// <summary>
        /// Every device ILGPU enumerated, with repeats collapsed to a count.
        /// <para>
        /// ILGPU offers one CPU accelerator per worker configuration, so this line read
        /// <c>CPU:CPUAccelerator | CPU:CPUAccelerator | ...</c> five times over on the development
        /// machine and the one entry that mattered was lost in it. Only IDENTICAL strings are collapsed,
        /// and the string is the whole of what this field ever printed - so nothing that was visible
        /// before is hidden now, and the multiplier keeps the total count readable.
        /// </para>
        /// </summary>
        private static string DescribeDevices(Context context)
        {
            var seen = new List<KeyValuePair<string, int>>();

            foreach (var device in context.Devices)
            {
                var name = $"{device.AcceleratorType}:{device.Name}";
                var at = seen.FindIndex(e => e.Key == name);

                if (at < 0)
                {
                    seen.Add(new KeyValuePair<string, int>(name, 1));
                    continue;
                }

                seen[at] = new KeyValuePair<string, int>(name, seen[at].Value + 1);
            }

            return string.Join(" | ", seen.Select(e => e.Value == 1 ? e.Key : $"{e.Key} x{e.Value}"));
        }

        /// <summary>
        /// Prints the parallel-unit count and the two numbers that bound occupancy with it, for whatever
        /// kind of accelerator this is.
        /// <para>
        /// <b>What the number IS, measured on 2026-08-22 rather than assumed.</b> ILGPU 1.5.3 declares
        /// <c>NumMultiprocessors</c> on the BASE <c>Device</c> type, and for an OpenCL device it carries
        /// the driver's <c>CL_DEVICE_MAX_COMPUTE_UNITS</c> unchanged. That was checked against the driver
        /// directly - <c>clGetDeviceInfo</c> on this machine's <c>gfx1036</c> returns 1 with status
        /// <c>CL_SUCCESS</c>, which is exactly what ILGPU reports, and the same query's
        /// <c>CL_DEVICE_MAX_CLOCK_FREQUENCY</c> of 2200 matches ILGPU's own <c>ClockRate</c>. So the value
        /// is a pass-through, not an ILGPU default.
        /// </para>
        /// <para>
        /// <b>What it is NOT.</b> It is the count the RUNTIME reports, which is not always the count of
        /// physical shader units - an OpenCL driver may report work-group processors rather than the
        /// cores inside them, and nothing this probe can observe distinguishes the two. The value is
        /// therefore labelled with the interface it came from, so a reader knows what to go and check
        /// rather than reading a hardware fact that was never measured.
        /// </para>
        /// <para>
        /// A non-positive count is printed as UNKNOWN with the raw value, never as a zero. "The runtime
        /// declined to answer" and "this device has no compute units" must not read the same.
        /// </para>
        /// </summary>
        private static void AddComputeUnits(Action<string, string> add, Accelerator accelerator)
        {
            var device = accelerator.Device;
            var units = device.NumMultiprocessors;
            var perUnit = device.MaxNumThreadsPerMultiprocessor;

            var source = accelerator.AcceleratorType switch
            {
                AcceleratorType.Cuda => "CUDA streaming multiprocessors, from cudaDeviceProp",
                AcceleratorType.OpenCL => "OpenCL CL_DEVICE_MAX_COMPUTE_UNITS, verbatim from the driver - " +
                                          "the runtime's count, which is not necessarily the count of " +
                                          "physical shader cores",
                AcceleratorType.CPU => "ILGPU's CPU emulator, which models a single multiprocessor",
                _ => "reported by ILGPU for this accelerator type",
            };

            add("accelerator compute units", units > 0
                ? $"{units} ({source})"
                : $"UNKNOWN - the runtime returned {units}, which is not a count. Read this as not " +
                  "answered, NOT as a device with no compute units.");

            add("accelerator max threads per compute unit", perUnit > 0
                ? perUnit.ToString()
                : $"UNKNOWN - the runtime returned {perUnit}");

            // Printed beside the two above so the arithmetic is visible rather than asserted. On both
            // devices seen here the product is exact: OpenCL gfx1036 gives 1 x 256 = 256, and the CPU
            // emulator gives 1 x 16 = 16. Whether ILGPU DERIVES one from the others was not checked.
            add("accelerator max threads total", device.MaxNumThreads.ToString());
        }

        public string Render()
        {
            var width = Fields.Max(kv => kv.Key.Length);
            var sb = new StringBuilder();
            foreach (var kv in Fields)
            {
                sb.Append("  ").Append(kv.Key.PadRight(width)).Append(" : ").AppendLine(kv.Value);
            }

            return sb.ToString();
        }

        private static string DescribeAssembly(Type type)
        {
            var assembly = type.Assembly;
            var name = assembly.GetName();

            var location = assembly.Location;
            var origin = location.Length == 0
                ? "a single-file or in-memory assembly"
                : location.Replace('\\', '/').Contains("/.nuget/packages/", StringComparison.OrdinalIgnoreCase)
                    ? "the NuGet cache"
                    : "beside the executable, copied there by the build";

            var fileVersion = "?";
            if (location.Length > 0)
            {
                fileVersion = FileVersionInfo.GetVersionInfo(location).FileVersion ?? "?";
            }

            return $"{name.Name} asm {name.Version} file {fileVersion} (resolved from {origin})";
        }

        private static string CpuModel()
        {
            if (X86Base.IsSupported)
            {
                var brand = X86BrandString();
                if (brand.Length > 0)
                {
                    return brand;
                }
            }

            if (OperatingSystem.IsLinux() && File.Exists("/proc/cpuinfo"))
            {
                foreach (var line in File.ReadLines("/proc/cpuinfo"))
                {
                    if (line.StartsWith("model name", StringComparison.OrdinalIgnoreCase))
                    {
                        return line[(line.IndexOf(':') + 1)..].Trim();
                    }
                }
            }

            return Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "unknown";
        }

        /// <summary>
        /// CPUID leaves 0x80000002-0x80000004 hold the 48-byte brand string. Read directly rather than
        /// from the Windows registry so the same code answers on Linux, and so the probe carries no
        /// platform-specific package.
        /// </summary>
        private static string X86BrandString()
        {
            var (maxExtended, _, _, _) = X86Base.CpuId(unchecked((int)0x80000000), 0);
            if ((uint)maxExtended < 0x80000004u)
            {
                return string.Empty;
            }

            Span<byte> buffer = stackalloc byte[48];
            var offset = 0;
            for (var leaf = 0x80000002; leaf <= 0x80000004; leaf++)
            {
                var (eax, ebx, ecx, edx) = X86Base.CpuId(unchecked((int)leaf), 0);
                foreach (var value in stackalloc[] { eax, ebx, ecx, edx })
                {
                    BitConverter.TryWriteBytes(buffer[offset..], value);
                    offset += 4;
                }
            }

            return Encoding.ASCII.GetString(buffer).TrimEnd('\0').Trim();
        }
    }
}
