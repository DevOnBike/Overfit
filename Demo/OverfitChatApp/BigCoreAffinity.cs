// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace DevOnBike.OverfitChat
{
    /// <summary>
    /// Pins every thread of this process to the SoC's fastest CPU cluster.
    /// <para>
    /// <b>Why this exists.</b> Measured on a Snapdragon 7s Gen 2 on 2026-08-14: during decode all four
    /// big cores sat at their 691 MHz idle floor while the four little cores ran near their 1.96 GHz
    /// ceiling — the whole model was running on the little cluster. The per-32-token rate series was
    /// <c>10.0 3.1 3.2 3.8 3.3 3.9 4.8 4.9 4.2</c>: the first window still rides the post-tap boost on a
    /// big core, then the scheduler settles the work onto the little cluster and leaves it there. The
    /// 2.5-3x A78-over-A55 gap is exactly the observed 10.0 to 3.1 step. The process is NOT confined by
    /// the system — its cpuset is <c>top-app</c> with cpus 0-7 allowed, no thermal throttling (30-39 C)
    /// and no memory reclaim (VmSwap 5 MB, zero major faults during generation). The cause is placement:
    /// each of the eight workers runs in short bursts and parks on a semaphore between dispatches, so no
    /// single thread ever accumulates the utilisation signal that would earn it a big core.
    /// </para>
    /// <para>
    /// <b>Cost.</b> Confining work to the big cluster spends more energy per token. Keep this only if the
    /// measured throughput justifies it, and prefer refining it (fewer, fatter workers; big+little rather
    /// than big-only) over leaving a blunt pin in place.
    /// </para>
    /// <para>
    /// Threads created after a call are unaffected, which is why the caller applies it again once decode
    /// has actually started — the engine's worker threads do not exist until the first dispatch.
    /// </para>
    /// </summary>
    internal static class BigCoreAffinity
    {
        // Linux takes a TID here: sched_setaffinity(0) means "the calling thread", and any TID from
        // /proc/self/task addresses that specific thread. There is no process-wide form.
        [DllImport("libc", SetLastError = true)]
        private static extern int sched_setaffinity(int threadId, IntPtr cpuSetSize, byte[] mask);

        // cpu_set_t is 1024 bits, and the kernel rejects a size smaller than the one it wants to fill.
        private const int CpuSetBytes = 128;

        /// <summary>
        /// Applies the pin to every thread that currently exists. Returns a one-line summary for the log —
        /// including failures, because a pin that silently did nothing looks exactly like a pin that did.
        /// </summary>
        public static string Apply()
        {
            var fastest = FastestCores();
            
            if (fastest.Count == 0)
            {
                return "skipped (no cpufreq data)";
            }

            var mask = new byte[CpuSetBytes];
            
            foreach (var core in fastest)
            {
                mask[core >> 3] |= (byte)(1 << (core & 7));
            }

            var pinned = 0;
            var failed = 0;
            
            foreach (var taskDir in System.IO.Directory.GetDirectories("/proc/self/task"))
            {
                if (!int.TryParse(System.IO.Path.GetFileName(taskDir), out var threadId))
                {
                    continue;
                }
                
                if (sched_setaffinity(threadId, (IntPtr)CpuSetBytes, mask) == 0)
                {
                    pinned++;
                    continue;
                }
                
                failed++;
            }

            return $"cores=[{string.Join(",", fastest)}] pinned={pinned} failed={failed}";
        }

        /// <summary>
        /// How many cores <see cref="Apply"/> would pin to, or 0 when the topology cannot be read. Callers
        /// use it to size a worker pool to the cluster the work will actually run on.
        /// </summary>
        public static int FastestCoreCount() => FastestCores().Count;

        /// <summary>
        /// The cores whose <c>cpuinfo_max_freq</c> equals the highest value on the machine. Read rather
        /// than hardcoded, so this stays correct on a different SoC (and on one with three clusters, where
        /// "big" is not simply "the top half of the core list").
        /// </summary>
        private static List<int> FastestCores()
        {
            var maxFrequencyByCore = new Dictionary<int, long>();
            
            for (var core = 0; core < 32; core++)
            {
                var path = $"/sys/devices/system/cpu/cpu{core}/cpufreq/cpuinfo_max_freq";
                
                if (!System.IO.File.Exists(path))
                {
                    continue;
                }
                
                if (long.TryParse(System.IO.File.ReadAllText(path).Trim(), out var frequency))
                {
                    maxFrequencyByCore[core] = frequency;
                }
            }

            var fastest = new List<int>();
            var highest = 0L;
            
            foreach (var entry in maxFrequencyByCore)
            {
                if (entry.Value > highest)
                {
                    highest = entry.Value;
                }
            }
            
            foreach (var entry in maxFrequencyByCore)
            {
                if (entry.Value == highest)
                {
                    fastest.Add(entry.Key);
                }
            }
            
            fastest.Sort();
            
            return fastest;
        }
    }
}
