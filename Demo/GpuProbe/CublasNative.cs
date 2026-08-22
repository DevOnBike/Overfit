// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Reflection;
using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The direct cuBLAS P/Invoke surface used by arm X3, and nothing else. It exists because
    /// <c>ILGPU.Algorithms 1.5.3</c> exports no <c>GemmEx</c> at all - verified against the assembly
    /// bytes on 2026-08-21: <c>cublasGemmEx</c> and <c>GemmEx</c> occur zero times, <c>cublasHgemm</c>
    /// and <c>cublasSgemm</c> once each - so the FP32-accumulate path a TENSOR CORE actually uses is
    /// unreachable through that wrapper.
    /// <para>
    /// <b>THE ABI CONSTANTS BELOW ARE VERIFIED, NOT BELIEVED.</b> They were read out of NVIDIA's own
    /// <c>cublas_api.h</c>, taken from the <c>nvidia-cublas-cu12</c> 12.9.2.10 win_amd64 wheel on
    /// 2026-08-22. The wheel is 553 MB and was NOT downloaded: a wheel is a zip, so HTTP range requests
    /// over its central directory pulled the single 381 kB header for about 100 kB of transfer. That
    /// matters because the design brief for this arm recorded these values as coming from two
    /// non-authoritative sources - another project's source and one person's recollection - and a wrong
    /// enum here produces a FAST WRONG NUMBER on a machine nobody here can see.
    /// </para>
    /// <para>
    /// <b>This is not a product interop layer and must never become one.</b> Decision D5 of the plan
    /// defers a P/Invoke GPU route for the product, and <c>Sources/**</c> contains zero
    /// <c>DllImport</c>, <c>LibraryImport</c> or <c>NativeLibrary</c>. This file lives in a throwaway
    /// probe outside <c>Overfit.sln</c>.
    /// </para>
    /// <para>
    /// <b>NOT ONE LINE OF THIS FILE HAS EVER CALLED CUBLAS.</b> The development machine has no NVIDIA
    /// device and no CUDA installation. Everything below is compile-checked, and the load path in
    /// <see cref="TryLoad"/> is the only part that runs here - where it correctly fails.
    /// </para>
    /// </summary>
    internal static partial class CublasNative
    {
        /// <summary>
        /// The name every <c>[LibraryImport]</c> below asks for. It is deliberately NOT a real file
        /// name: <see cref="Resolve"/> maps it to whichever versioned cuBLAS this machine actually has.
        /// </summary>
        private const string LibraryName = "cublas";

        /// <summary>
        /// Windows candidates, most recent first. <b>Trying 13 is the one capability arm X3 has that
        /// arms X1 and X2 do not.</b> ILGPU 1.5.3's string table contains <c>cublas64_10</c>,
        /// <c>cublas64_11</c> and <c>cublas64_12</c> and no name for a major 13, so a machine carrying
        /// only a CUDA 13 redistributable skips both ILGPU cuBLAS arms while X3 can still run.
        /// <para>
        /// That <c>cublas64_13.dll</c> is the correct name for CUDA 13 on Windows is NOT verified here.
        /// It follows the naming of every previous major and it is what dotLLM's resolver tries; there
        /// is no CUDA installation on this machine to check it against.
        /// </para>
        /// </summary>
        private static readonly string[] WindowsCandidates =
            ["cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll"];

        /// <summary>
        /// Linux candidates. The bare <c>libcublas.so</c> is last and only helps where a development
        /// package installed the unversioned symlink.
        /// </summary>
        private static readonly string[] UnixCandidates =
            ["libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so"];

        private static readonly object Gate = new();

        private static nint _module;
        private static bool _resolverRegistered;

        /// <summary>CUBLAS_OP_N. Verified: <c>cublas_api.h</c>, <c>cublasOperation_t</c>.</summary>
        public const int OpNonTranspose = 0;

        /// <summary>CUBLAS_OP_T. Verified: <c>cublas_api.h</c>, <c>cublasOperation_t</c>.</summary>
        public const int OpTranspose = 1;

        /// <summary>CUDA_R_16F. Verified: <c>library_types.h</c>, <c>cudaDataType_t</c>.</summary>
        public const int DataTypeR16F = 2;

        /// <summary>CUDA_R_32F. Verified: <c>library_types.h</c>, <c>cudaDataType_t</c>.</summary>
        public const int DataTypeR32F = 0;

        /// <summary>
        /// CUBLAS_COMPUTE_32F. Verified: <c>cublas_api.h</c>, <c>cublasComputeType_t</c>, commented
        /// <c>/* float - default */</c>. <b>This is the whole point of arm X3.</b>
        /// </summary>
        public const int Compute32F = 68;

        /// <summary>
        /// CUBLAS_COMPUTE_16F. Verified: <c>cublas_api.h</c>. Declared here but never passed, on
        /// purpose: it is the single most plausible mistake in this arm - the neighbouring value that
        /// would silently give FP16 accumulate, which measures 6.5e-3 relative L2 at k=2048 against a
        /// 1e-3 parity ceiling. Naming it makes the wrong constant visible instead of a bare 68 that a
        /// reader has to take on trust.
        /// </summary>
        public const int Compute16F = 64;

        /// <summary>
        /// CUBLAS_GEMM_DEFAULT. Verified: <c>cublas_api.h</c>, <c>cublasGemmAlgo_t</c>, where
        /// <c>CUBLAS_GEMM_DFALT</c> and <c>CUBLAS_GEMM_DEFAULT</c> are both -1.
        /// </summary>
        public const int GemmDefault = -1;

        /// <summary>CUBLAS_STATUS_SUCCESS. Verified: <c>cublas_api.h</c>, <c>cublasStatus_t</c>.</summary>
        public const int StatusSuccess = 0;

        /// <summary>
        /// The library name that actually loaded, or null before <see cref="TryLoad"/> succeeds. It is
        /// an OBSERVATION of what the loader took, not the first name in the candidate list.
        /// </summary>
        public static string? ResolvedName { get; private set; }

        /// <summary>The candidate names for this operating system, in the order they are tried.</summary>
        public static string[] Candidates => OperatingSystem.IsWindows() ? WindowsCandidates : UnixCandidates;

        /// <summary>
        /// Forces the native library to resolve NOW and reports what happened, rather than letting the
        /// first call decide it.
        /// <para>
        /// <b>This method exists because the failure is LAZY, and that was measured rather than
        /// assumed.</b> A missing <c>cublas64_*.dll</c> does not surface at type load and does not
        /// surface when the resolver is registered: it surfaces as a <see cref="DllNotFoundException"/>
        /// at the FIRST CALL, identically under Native-AOT and the JIT. So merely constructing the arm
        /// proves nothing about whether it can run, and an arm that reported itself available on that
        /// basis would throw in the middle of a measurement.
        /// </para>
        /// <para>
        /// <b>Resolution is by bare name and never by a full path</b>, so that a module already mapped
        /// into this process satisfies it. That is what keeps X3 and ILGPU's cuBLAS wrapper on the same
        /// loaded library when both are present.
        /// </para>
        /// </summary>
        /// <param name="reason">
        /// Empty on success. On failure, a STABLE ENGLISH sentence naming every candidate tried, with
        /// the operating system's own text appended as a quotation. The OS text is localised - on the
        /// development machine it comes back in Polish - so it cannot be the only thing a reader gets.
        /// </param>
        public static bool TryLoad(out string reason)
        {
            lock (Gate)
            {
                if (_module != nint.Zero)
                {
                    reason = string.Empty;
                    return true;
                }

                EnsureResolver();

                // BOUND: at most Candidates.Length iterations - a fixed 3- or 4-element array above.
                foreach (var name in Candidates)
                {
                    if (NativeLibrary.TryLoad(name, out var handle))
                    {
                        // The handle is kept for the process lifetime on purpose. Freeing it would drop
                        // this reference while every P/Invoke below still needs the module mapped.
                        _module = handle;
                        ResolvedName = name;
                        reason = string.Empty;
                        return true;
                    }
                }

                reason = "X3: no cuBLAS library could be loaded. " + DescribeFailures() +
                         " Install the CUDA redistributable, or check that its directory is on the " +
                         "library search path.";
                return false;
            }
        }

        /// <summary>
        /// Asks the loader again for EVERY candidate, one at a time, only to obtain the operating
        /// system's own message for each - <see cref="NativeLibrary.TryLoad(string, out nint)"/> returns
        /// false and no text. Each message is quoted rather than presented as the reason, because the OS
        /// localises it.
        /// <para>
        /// <b>Every candidate, not just the last one, and that was a correction rather than a
        /// preference.</b> The first version reported only the final name in the list. A machine where
        /// <c>cublas64_12.dll</c> exists but is corrupt or 32-bit would then have been told that
        /// <c>cublas64_11.dll</c> was missing - true, useless, and pointing at the wrong file. A wrong
        /// diagnosis that reads like a right one is the failure mode this whole probe is built to avoid,
        /// and it costs one extra load attempt on a path that has already failed.
        /// </para>
        /// </summary>
        private static string DescribeFailures()
        {
            var parts = new List<string>();

            // BOUND: at most Candidates.Length iterations - a fixed 3- or 4-element array.
            foreach (var name in Candidates)
            {
                parts.Add(name + ": " + DescribeFailure(name));
            }

            return "Tried, in order - " + string.Join("; ", parts) +
                   ". The quoted text is the operating system's own, in the machine's language, which " +
                   "may not be English.";
        }

        private static string DescribeFailure(string name)
        {
            try
            {
                var handle = NativeLibrary.Load(name);
                NativeLibrary.Free(handle);
                return "the loader accepted it on a second attempt, which should not happen - treat this " +
                       "run's X3 result with suspicion";
            }
            catch (DllNotFoundException ex)
            {
                return $"not found, \"{ex.Message.Trim()}\"";
            }
            catch (BadImageFormatException ex)
            {
                return "PRESENT BUT NOT LOADABLE - a 32-bit library in a 64-bit process, or a truncated " +
                       $"download, gives exactly this: \"{ex.Message.Trim()}\"";
            }
        }

        /// <summary>
        /// Registers the resolver that maps the bare name <c>cublas</c> onto whichever versioned library
        /// this machine has. Registration itself loads nothing and proves nothing.
        /// </summary>
        private static void EnsureResolver()
        {
            if (_resolverRegistered)
            {
                return;
            }

            NativeLibrary.SetDllImportResolver(typeof(CublasNative).Assembly, Resolve);
            _resolverRegistered = true;
        }

        private static nint Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName != LibraryName)
            {
                return nint.Zero;
            }

            return _module;
        }

        /// <summary>
        /// Turns a <c>cublasStatus_t</c> into its documented name. A bare number in a report from
        /// somebody else's machine is not something the reader can act on, and the word "failed" is
        /// worse - <c>CUBLAS_STATUS_ARCH_MISMATCH</c> and <c>CUBLAS_STATUS_ALLOC_FAILED</c> lead to
        /// completely different next steps. Values verified against <c>cublasStatus_t</c>; note the
        /// enum is deliberately not contiguous.
        /// </summary>
        public static string StatusName(int status) => status switch
        {
            0 => "CUBLAS_STATUS_SUCCESS",
            1 => "CUBLAS_STATUS_NOT_INITIALIZED",
            3 => "CUBLAS_STATUS_ALLOC_FAILED",
            7 => "CUBLAS_STATUS_INVALID_VALUE",
            8 => "CUBLAS_STATUS_ARCH_MISMATCH",
            11 => "CUBLAS_STATUS_MAPPING_ERROR",
            13 => "CUBLAS_STATUS_EXECUTION_FAILED",
            14 => "CUBLAS_STATUS_INTERNAL_ERROR",
            15 => "CUBLAS_STATUS_NOT_SUPPORTED",
            16 => "CUBLAS_STATUS_LICENSE_ERROR",
            _ => "an unrecognised cublasStatus_t",
        };

        /// <summary>
        /// Renders a cuBLAS version integer the way NVIDIA composes it. Verified, not recalled:
        /// <c>cublas_api.h</c> line 90 defines <c>CUBLAS_VERSION</c> as
        /// <c>CUBLAS_VER_MAJOR * 10000 + CUBLAS_VER_MINOR * 100 + CUBLAS_VER_PATCH</c>, and the same
        /// header's 12/9/2 triple renders as 12.9.2 from 120902. Printed beside ILGPU's own version so a
        /// disagreement - which would mean X1 and X3 measured two different libraries - is visible.
        /// </summary>
        public static string DescribeVersion(int version) =>
            $"{version / 10000}.{version / 100 % 100}.{version % 100} (raw {version})";

        /// <summary>Creates a cuBLAS handle owned by this process. <c>cublasCreate_v2</c>.</summary>
        [LibraryImport(LibraryName, EntryPoint = "cublasCreate_v2")]
        internal static partial int Create(out nint handle);

        /// <summary>Destroys a handle created by <see cref="Create"/>. <c>cublasDestroy_v2</c>.</summary>
        [LibraryImport(LibraryName, EntryPoint = "cublasDestroy_v2")]
        internal static partial int Destroy(nint handle);

        /// <summary>
        /// Binds the handle to a CUDA stream. <c>cublasSetStream_v2</c>. X3 passes ILGPU's default
        /// stream so that the existing <c>accelerator.Synchronize()</c> barrier in <see cref="Arm"/>
        /// really does wait for this work.
        /// </summary>
        [LibraryImport(LibraryName, EntryPoint = "cublasSetStream_v2")]
        internal static partial int SetStream(nint handle, nint stream);

        /// <summary>
        /// Reports the loaded library's version. <c>cublasGetVersion_v2</c>. A DIAGNOSTIC only - X3 owns
        /// its handle, so a disagreement with ILGPU does not invalidate the run; it means the report
        /// must not present X1 and X3 as two measurements of one library.
        /// </summary>
        [LibraryImport(LibraryName, EntryPoint = "cublasGetVersion_v2")]
        internal static partial int GetVersion(nint handle, out int version);

        /// <summary>
        /// <c>cublasGemmEx</c>. The signature is transcribed from <c>cublas_api.h</c>:
        /// <c>transa</c>/<c>transb</c> are <c>cublasOperation_t</c>, the three type parameters are
        /// <c>cudaDataType</c>, <c>computeType</c> is <c>cublasComputeType_t</c> and <c>algo</c> is
        /// <c>cublasGemmAlgo_t</c> - all C enums, so all 4-byte ints.
        /// <para>
        /// <c>alpha</c> and <c>beta</c> are <c>const void*</c> in the header and their pointee type is
        /// decided by <c>computeType</c>: with <see cref="Compute32F"/> they are <c>float</c>. They are
        /// HOST pointers, because cuBLAS defaults to <c>CUBLAS_POINTER_MODE_HOST</c> and X3 owns its
        /// handle and never changes that. Declaring them <c>in float</c> lets the marshaller take the
        /// address of a caller-held field, which allocates nothing inside a timed region and needs no
        /// hand-written pointer arithmetic. It does NOT avoid <c>AllowUnsafeBlocks</c>: measured on
        /// 2026-08-22, <c>[LibraryImport]</c> emits its stubs inside <c>unsafe</c> blocks whatever the
        /// parameter shapes are, and the build fails with <c>SYSLIB1062</c> without the flag.
        /// </para>
        /// </summary>
        [LibraryImport(LibraryName, EntryPoint = "cublasGemmEx")]
        internal static partial int GemmEx(
            nint handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            in float alpha,
            nint a,
            int aType,
            int lda,
            nint b,
            int bType,
            int ldb,
            in float beta,
            nint c,
            int cType,
            int ldc,
            int computeType,
            int algo);
    }
}
