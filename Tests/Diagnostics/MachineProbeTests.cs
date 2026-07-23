// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Diagnostics;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Diagnostics
{
    /// <summary>
    /// Measures this machine's hardware ceilings — peak FMA throughput, sustained memory bandwidth, and
    /// bandwidth versus working-set size — so every performance number elsewhere can be read as a fraction of
    /// what the box can actually do. A bare "1.7 TFLOP/s" means nothing until you know whether the machine
    /// tops out at 2 or at 20.
    ///
    /// <para><b>Two method traps this file exists to avoid, both of which this project has fallen into.</b></para>
    /// <list type="number">
    ///   <item><b>Accumulators must be named locals</b>, never a <c>stackalloc</c> span: a span forces an L1
    ///     round-trip per accumulator per iteration and measures cache latency rather than issue rate. That
    ///     mistake once reported a float peak of 0.79 TFLOP/s — below what a real matmul achieved, which is
    ///     impossible for a loop that touches no memory.</item>
    ///   <item><b>Several independent chains, never one.</b> A single accumulator measures the dependency
    ///     chain's latency, not throughput. An earlier version of the working-set sweep used one
    ///     <c>Vector&lt;float&gt;</c> accumulator and produced a perfectly flat ~75 GB/s from L1 to DRAM — not
    ///     because the machine has no cache cliff, but because 75 GB/s was the latency ceiling of that chain,
    ///     below every cache level's bandwidth. The flat curve was an artefact, and the conclusion drawn from
    ///     it ("this CPU has no cache cliff, so blocking cannot pay") was withdrawn.</item>
    /// </list>
    ///
    /// <para>A multiply-accumulate counts as 2 operations, matching <see cref="Throughput"/> and llama.cpp's
    /// <c>test-backend-ops</c>, so figures are directly comparable across the two projects. Every number is a
    /// best-of-N: noise only ever makes a machine look slower.</para>
    ///
    /// <para><see cref="ValueStopwatch"/> rather than <see cref="System.Diagnostics.Stopwatch"/> throughout —
    /// the allocation-free timer this repo standardises on.</para>
    /// </summary>
    public sealed class MachineProbeTests
    {
        /// <summary>Independent accumulator chains — enough to hide FMA latency without exhausting registers.</summary>
        private const int Chains = 12;

        /// <summary>Independent load streams in the bandwidth loops, for the same reason.</summary>
        private const int Streams = 8;

        private const int ComputeIterations = 2_000_000;
        private const int Repeats = 5;

        /// <summary>256 MB — must dwarf any last-level cache, including a 128 MB V-cache.</summary>
        private const int MemoryFloats = 64 * 1024 * 1024;

        private static float _sink;

        private readonly ITestOutputHelper _out;

        public MachineProbeTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Probe_HardwareCeilings()
        {
            var cores = Environment.ProcessorCount;

            _out.WriteLine("=== machine ceilings ===");
            _out.WriteLine($"  CPU          : {ReadCpuName()}");
            _out.WriteLine($"  clock        : {MeasureClockGhz(),0:F2} GHz (measured, not reported)");
            _out.WriteLine($"  logical CPUs : {cores}");
            _out.WriteLine($"  vector width : {Vector<float>.Count * 32} bit ({Vector<float>.Count} floats)");
            _out.WriteLine($"  ISA          : {DescribeIsa()}");
            _out.WriteLine($"  cache        : {ReadCacheTopology()}");
            _out.WriteLine(string.Empty);

            ReportCompute(cores);
            _out.WriteLine(string.Empty);
            ReportMemory(cores);
            _out.WriteLine(string.Empty);
            ReportWorkingSetSweep(cores);
            _out.WriteLine(string.Empty);
            ReportAllocations();

            Assert.True(cores > 0);
        }

        /// <summary>
        /// Confirms the measured loops themselves allocate nothing, so the figures above are not partly a GC
        /// measurement.
        ///
        /// <para>The buffers are allocated during setup and that is unavoidable; what must be zero is the
        /// <i>timed</i> region. The parallel arms are reported separately and are <b>not</b> expected to be
        /// zero — <see cref="Parallel.For(int, int, Action{int})"/> allocates its own state per call — which is
        /// exactly why the single-core figures are the ones to trust for a clean rate.</para>
        /// </summary>
        private void ReportAllocations()
        {
            _out.WriteLine("--- allocation check on the measured loops ---");

            var data = new float[1024 * 1024];
            for (var i = 0; i < data.Length; i++)
            {
                data[i] = i;
            }

            _out.WriteLine($"  FMA chain (256-bit)  {AllocatedBy(() => FmaChains256()),8} B");
            _out.WriteLine($"  read loop            {AllocatedBy(() => ReadRepeated(data, 0, data.Length, 4)),8} B");

            var cores = Environment.ProcessorCount;
            _out.WriteLine($"  parallel read        {AllocatedBy(() => ParallelRead(data, cores)),8} B"
                + "   (Parallel.For state — expected, not part of any reported rate)");

            var single = AllocatedBy(() => ReadRepeated(data, 0, data.Length, 4));
            var fma = AllocatedBy(() => FmaChains256());

            Assert.True(single == 0, $"read loop allocated {single} B — the sweep would be measuring GC, not bandwidth");
            Assert.True(fma == 0, $"FMA chain allocated {fma} B — the peak figures would include GC work");
        }

        private static long AllocatedBy(Func<float> body)
        {
            body();
            GC.Collect();
            GC.WaitForPendingFinalizers();

            var before = GC.GetAllocatedBytesForCurrentThread();
            _sink = body();

            return GC.GetAllocatedBytesForCurrentThread() - before;
        }

        /// <summary>
        /// Core clock, <b>measured</b> rather than read from a spec sheet: a chain of dependent integer adds
        /// retires exactly one per cycle on every mainstream core, so iterations per second is the clock.
        /// Reporting the measured value matters because boost, thermal and power state make the nameplate
        /// figure wrong most of the time — and every GF/s number above has to be divided by the clock that was
        /// actually in effect.
        /// </summary>
        private static double MeasureClockGhz()
        {
            const int Iterations = 200_000_000;

            var best = double.MaxValue;

            for (var r = 0; r < 3; r++)
            {
                var x = 1;
                var started = ValueStopwatch.StartNew();

                for (var i = 0; i < Iterations; i++)
                {
                    x += x & 1; // dependent on the previous value: one add per cycle
                }

                var seconds = started.GetElapsedTime().TotalSeconds;
                _sink = x;
                best = Math.Min(best, seconds);
            }

            // Two dependent ops per iteration (the AND and the ADD) plus loop overhead the JIT folds away;
            // treat this as a lower bound on the true clock rather than a precise figure.
            return Iterations * 2.0 / best / 1e9;
        }

        /// <summary>CPU model from the OS, best-effort and non-fatal — the measurements stand without it.</summary>
        private static string ReadCpuName()
        {
            try
            {
                if (OperatingSystem.IsWindows())
                {
                    var value = Microsoft.Win32.Registry.GetValue(
                        @"HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                        "ProcessorNameString", null);

                    return value?.ToString()?.Trim() ?? "(unknown)";
                }

                if (OperatingSystem.IsLinux() && File.Exists("/proc/cpuinfo"))
                {
                    foreach (var line in File.ReadLines("/proc/cpuinfo"))
                    {
                        if (line.StartsWith("model name", StringComparison.Ordinal))
                        {
                            return line[(line.IndexOf(':') + 1)..].Trim();
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                return $"(unavailable: {ex.GetType().Name})";
            }

            return "(unknown)";
        }

        /// <summary>
        /// Cache sizes per level, best-effort from the OS. Where it is unavailable the working-set sweep below
        /// still shows the effective boundaries, which is the number that actually governs a kernel — reported
        /// capacity and usable capacity are not the same thing on a CPU with stacked cache.
        /// </summary>
        private static string ReadCacheTopology()
        {
            try
            {
                if (OperatingSystem.IsLinux())
                {
                    var parts = new List<string>();

                    for (var index = 0; index < 6; index++)
                    {
                        var levelPath = $"/sys/devices/system/cpu/cpu0/cache/index{index}/level";
                        var sizePath = $"/sys/devices/system/cpu/cpu0/cache/index{index}/size";

                        if (!File.Exists(levelPath) || !File.Exists(sizePath))
                        {
                            continue;
                        }

                        parts.Add($"L{File.ReadAllText(levelPath).Trim()}={File.ReadAllText(sizePath).Trim()}");
                    }

                    if (parts.Count > 0)
                    {
                        return string.Join(" ", parts);
                    }
                }
            }
            catch (Exception ex)
            {
                return $"(unavailable: {ex.GetType().Name})";
            }

            return "(see the working-set sweep below — effective sizes are measured there)";
        }

        private static string DescribeIsa()
        {
            var parts = new List<string>();

            if (Avx2.IsSupported)
            {
                parts.Add("AVX2");
            }
            if (Fma.IsSupported)
            {
                parts.Add("FMA");
            }
            if (Avx512F.IsSupported)
            {
                parts.Add("AVX512F");
            }
            if (Avx512BW.IsSupported)
            {
                parts.Add("AVX512BW");
            }
            if (AdvSimd.IsSupported)
            {
                parts.Add("NEON");
            }

            return parts.Count == 0 ? "(baseline)" : string.Join(" ", parts);
        }

        private void ReportCompute(int cores)
        {
            _out.WriteLine("--- peak FMA (register-resident, no memory traffic) ---");
            _out.WriteLine($"  {"width",-10}{"1 core",13}{"all cores",14}{"scaling",10}");

            MeasureCompute("128-bit", FmaChains128, Vector128<float>.Count, cores);

            if (Avx.IsSupported && Fma.IsSupported)
            {
                MeasureCompute("256-bit", FmaChains256, Vector256<float>.Count, cores);
            }

            if (Avx512F.IsSupported)
            {
                MeasureCompute("512-bit", FmaChains512, Vector512<float>.Count, cores);
            }
        }

        private void MeasureCompute(string label, Func<float> body, int lanes, int cores)
        {
            var flopsPerCall = 2.0 * Chains * ComputeIterations * lanes;

            var single = BestSeconds(body);
            var all = BestSeconds(() =>
            {
                var partial = new float[cores];
                Parallel.For(0, cores, new ParallelOptions { MaxDegreeOfParallelism = cores },
                    i => partial[i] = body());
                return partial[0];
            });

            var oneCore = flopsPerCall / single / 1e9;
            var allCores = flopsPerCall * cores / all / 1e9;

            _out.WriteLine($"  {label,-10}{oneCore,8:F0} GF/s{allCores,9:F0} GF/s{allCores / oneCore,9:F1}x");
        }

        private void ReportMemory(int cores)
        {
            _out.WriteLine("--- sustained bandwidth, 256 MB buffers (past any cache) ---");

            var x = new float[MemoryFloats];
            var y = new float[MemoryFloats];
            var z = new float[MemoryFloats];
            var rng = new Random(20260723);

            for (var i = 0; i < MemoryFloats; i++)
            {
                x[i] = (float)rng.NextDouble();
                y[i] = (float)rng.NextDouble();
                z[i] = (float)rng.NextDouble();
            }

            const long Bytes = (long)MemoryFloats * sizeof(float);

            var oneCoreRead = Bytes / BestSeconds(() => ReadRange(x, 0, x.Length));
            var allCoreRead = Bytes / BestSeconds(() => ParallelRead(x, cores));

            _out.WriteLine($"  read  1 core  {oneCoreRead / 1e9,8:F1} GB/s");
            _out.WriteLine($"  read  all     {allCoreRead / 1e9,8:F1} GB/s   {allCoreRead / oneCoreRead,5:F2}x scaling");
            _out.WriteLine($"  copy  all     {2 * Bytes / BestSeconds(() => { ParallelCopy(x, y, cores); return y[0]; }) / 1e9,8:F1} GB/s");
            _out.WriteLine($"  triad all     {3 * Bytes / BestSeconds(() => { ParallelTriad(x, y, z, cores); return x[0]; }) / 1e9,8:F1} GB/s");
        }

        /// <summary>
        /// Read bandwidth against working-set size, at one core and at all cores, using <see cref="Streams"/>
        /// independent accumulators so the loop is throughput-bound rather than latency-bound.
        ///
        /// <para><b>The all-core column is the one that explains parallel scaling.</b> Private L1/L2 scale with
        /// cores; a shared L3 and DRAM do not. Comparing the two columns at each size shows exactly where
        /// adding cores stops buying bandwidth — which is the difference between "this kernel is slow" and
        /// "this kernel is fed slowly", and no compute measurement can distinguish them.</para>
        ///
        /// <para>Each core reads its <b>own private buffer</b> in the all-core arm, not a shared one: sharing
        /// would measure cache-line replication rather than aggregate bandwidth.</para>
        /// </summary>
        private void ReportWorkingSetSweep(int cores)
        {
            _out.WriteLine("--- read bandwidth vs working set (independent streams) ---");
            _out.WriteLine($"  {"size",9}{"1 core",12}{"all cores",13}{"scaling",10}");

            int[] kilobytes = [16, 32, 48, 64, 128, 256, 512, 1024, 2048, 8192, 32768, 131072];

            foreach (var kb in kilobytes)
            {
                var floats = kb * 1024 / sizeof(float);
                var passes = Math.Max(8, (int)(64L * 1024 * 1024 / ((long)floats * sizeof(float))));

                var single = new float[floats];
                for (var i = 0; i < floats; i++)
                {
                    single[i] = i;
                }

                var oneCore = (double)floats * sizeof(float) * passes
                    / BestSeconds(() => ReadRepeated(single, 0, floats, passes)) / 1e9;

                // One private buffer per worker: a shared buffer would measure replication, not bandwidth.
                var buffers = new float[cores][];
                for (var w = 0; w < cores; w++)
                {
                    buffers[w] = new float[floats];
                    Array.Copy(single, buffers[w], floats);
                }

                var allSeconds = BestSeconds(() =>
                {
                    var partial = new float[cores];
                    Parallel.For(0, cores, new ParallelOptions { MaxDegreeOfParallelism = cores },
                        w => partial[w] = ReadRepeated(buffers[w], 0, floats, passes));
                    return partial[0];
                });

                var allCores = (double)floats * sizeof(float) * passes * cores / allSeconds / 1e9;

                _out.WriteLine(
                    $"  {kb,6} KB {oneCore,8:F1} GB/s{allCores,9:F1} GB/s{allCores / oneCore,8:F1}x  "
                    + new string('#', Math.Min(40, (int)(allCores / 25))));
            }
        }

        /// <summary>
        /// Sums a range <paramref name="passes"/> times through <see cref="Streams"/> independent accumulators.
        ///
        /// <para>The pass loop lives <i>inside</i> the accumulator setup deliberately. With the repetition
        /// outside, a small window's cost is dominated by re-initialising eight accumulators and doing a
        /// horizontal reduction per call — which is why an earlier version reported 7.5 GB/s at 32 KB, below
        /// its own DRAM figure. That was measurement overhead, not bandwidth.</para>
        /// </summary>
        private static float ReadRepeated(float[] data, int from, int to, int passes)
        {
            var width = Vector<float>.Count;
            var step = width * Streams;

            Vector<float> a0 = default, a1 = default, a2 = default, a3 = default;
            Vector<float> a4 = default, a5 = default, a6 = default, a7 = default;

            for (var p = 0; p < passes; p++)
            {
                var i = from;
                for (; i <= to - step; i += step)
                {
                    a0 += new Vector<float>(data.AsSpan(i, width));
                    a1 += new Vector<float>(data.AsSpan(i + width, width));
                    a2 += new Vector<float>(data.AsSpan(i + (2 * width), width));
                    a3 += new Vector<float>(data.AsSpan(i + (3 * width), width));
                    a4 += new Vector<float>(data.AsSpan(i + (4 * width), width));
                    a5 += new Vector<float>(data.AsSpan(i + (5 * width), width));
                    a6 += new Vector<float>(data.AsSpan(i + (6 * width), width));
                    a7 += new Vector<float>(data.AsSpan(i + (7 * width), width));
                }
            }

            return Vector.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
        }

        /// <summary>Single pass over a range — used by the whole-buffer bandwidth figures.</summary>
        private static float ReadRange(float[] data, int from, int to)
        {
            return ReadRepeated(data, from, to, 1);
        }

        private static float ParallelRead(float[] source, int cores)
        {
            var partial = new float[cores];

            Parallel.For(0, cores, new ParallelOptions { MaxDegreeOfParallelism = cores }, w =>
            {
                var (from, to) = Slice(source.Length, cores, w);
                partial[w] = ReadRange(source, from, to);
            });

            var total = 0f;
            for (var i = 0; i < partial.Length; i++)
            {
                total += partial[i];
            }

            return total;
        }

        private static void ParallelCopy(float[] source, float[] destination, int cores)
        {
            Parallel.For(0, cores, new ParallelOptions { MaxDegreeOfParallelism = cores }, w =>
            {
                var (from, to) = Slice(source.Length, cores, w);
                source.AsSpan(from, to - from).CopyTo(destination.AsSpan(from, to - from));
            });
        }

        private static void ParallelTriad(float[] a, float[] b, float[] c, int cores)
        {
            Parallel.For(0, cores, new ParallelOptions { MaxDegreeOfParallelism = cores }, w =>
            {
                var (from, to) = Slice(a.Length, cores, w);
                var scalar = new Vector<float>(3f);
                var width = Vector<float>.Count;
                var i = from;

                for (; i <= to - width; i += width)
                {
                    var vb = new Vector<float>(b.AsSpan(i, width));
                    var vc = new Vector<float>(c.AsSpan(i, width));
                    (vb + (scalar * vc)).CopyTo(a.AsSpan(i, width));
                }

                for (; i < to; i++)
                {
                    a[i] = b[i] + (3f * c[i]);
                }
            });
        }

        private static (int From, int To) Slice(int length, int workers, int worker)
        {
            var width = Vector<float>.Count;
            var per = length / workers / width * width;
            var from = worker * per;

            return (from, worker == workers - 1 ? length : from + per);
        }

        private static double BestSeconds(Func<float> body)
        {
            body();

            var best = double.MaxValue;

            for (var i = 0; i < Repeats; i++)
            {
                var started = ValueStopwatch.StartNew();
                _sink = body();
                var elapsed = started.GetElapsedTime().TotalSeconds;
                best = Math.Min(best, elapsed);
            }

            return best;
        }

        private static float FmaChains128()
        {
            var m = Vector128.Create(1.000001f);
            var a = Vector128.Create(0.000001f);

            Vector128<float> c0 = Vector128.Create(1f), c1 = Vector128.Create(2f);
            Vector128<float> c2 = Vector128.Create(3f), c3 = Vector128.Create(4f);
            Vector128<float> c4 = Vector128.Create(5f), c5 = Vector128.Create(6f);
            Vector128<float> c6 = Vector128.Create(7f), c7 = Vector128.Create(8f);
            Vector128<float> c8 = Vector128.Create(9f), c9 = Vector128.Create(10f);
            Vector128<float> c10 = Vector128.Create(11f), c11 = Vector128.Create(12f);

            for (var i = 0; i < ComputeIterations; i++)
            {
                c0 = (c0 * m) + a;
                c1 = (c1 * m) + a;
                c2 = (c2 * m) + a;
                c3 = (c3 * m) + a;
                c4 = (c4 * m) + a;
                c5 = (c5 * m) + a;
                c6 = (c6 * m) + a;
                c7 = (c7 * m) + a;
                c8 = (c8 * m) + a;
                c9 = (c9 * m) + a;
                c10 = (c10 * m) + a;
                c11 = (c11 * m) + a;
            }

            return Vector128.Sum(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11);
        }

        private static float FmaChains256()
        {
            var m = Vector256.Create(1.000001f);
            var a = Vector256.Create(0.000001f);

            Vector256<float> c0 = Vector256.Create(1f), c1 = Vector256.Create(2f);
            Vector256<float> c2 = Vector256.Create(3f), c3 = Vector256.Create(4f);
            Vector256<float> c4 = Vector256.Create(5f), c5 = Vector256.Create(6f);
            Vector256<float> c6 = Vector256.Create(7f), c7 = Vector256.Create(8f);
            Vector256<float> c8 = Vector256.Create(9f), c9 = Vector256.Create(10f);
            Vector256<float> c10 = Vector256.Create(11f), c11 = Vector256.Create(12f);

            for (var i = 0; i < ComputeIterations; i++)
            {
                c0 = Fma.MultiplyAdd(c0, m, a);
                c1 = Fma.MultiplyAdd(c1, m, a);
                c2 = Fma.MultiplyAdd(c2, m, a);
                c3 = Fma.MultiplyAdd(c3, m, a);
                c4 = Fma.MultiplyAdd(c4, m, a);
                c5 = Fma.MultiplyAdd(c5, m, a);
                c6 = Fma.MultiplyAdd(c6, m, a);
                c7 = Fma.MultiplyAdd(c7, m, a);
                c8 = Fma.MultiplyAdd(c8, m, a);
                c9 = Fma.MultiplyAdd(c9, m, a);
                c10 = Fma.MultiplyAdd(c10, m, a);
                c11 = Fma.MultiplyAdd(c11, m, a);
            }

            return Vector256.Sum(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11);
        }

        private static float FmaChains512()
        {
            var m = Vector512.Create(1.000001f);
            var a = Vector512.Create(0.000001f);

            Vector512<float> c0 = Vector512.Create(1f), c1 = Vector512.Create(2f);
            Vector512<float> c2 = Vector512.Create(3f), c3 = Vector512.Create(4f);
            Vector512<float> c4 = Vector512.Create(5f), c5 = Vector512.Create(6f);
            Vector512<float> c6 = Vector512.Create(7f), c7 = Vector512.Create(8f);
            Vector512<float> c8 = Vector512.Create(9f), c9 = Vector512.Create(10f);
            Vector512<float> c10 = Vector512.Create(11f), c11 = Vector512.Create(12f);

            for (var i = 0; i < ComputeIterations; i++)
            {
                c0 = Avx512F.FusedMultiplyAdd(c0, m, a);
                c1 = Avx512F.FusedMultiplyAdd(c1, m, a);
                c2 = Avx512F.FusedMultiplyAdd(c2, m, a);
                c3 = Avx512F.FusedMultiplyAdd(c3, m, a);
                c4 = Avx512F.FusedMultiplyAdd(c4, m, a);
                c5 = Avx512F.FusedMultiplyAdd(c5, m, a);
                c6 = Avx512F.FusedMultiplyAdd(c6, m, a);
                c7 = Avx512F.FusedMultiplyAdd(c7, m, a);
                c8 = Avx512F.FusedMultiplyAdd(c8, m, a);
                c9 = Avx512F.FusedMultiplyAdd(c9, m, a);
                c10 = Avx512F.FusedMultiplyAdd(c10, m, a);
                c11 = Avx512F.FusedMultiplyAdd(c11, m, a);
            }

            return Vector512.Sum(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11);
        }
    }
}
