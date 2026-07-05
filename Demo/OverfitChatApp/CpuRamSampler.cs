// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.OverfitChat
{
    /// <summary>
    /// Samples this process's live CPU usage (cores busy) and resident memory (RSS) from <c>/proc/self/*</c>
    /// — to show "look, it's running on your phone" stats while the model generates. CPU is a delta between
    /// calls (jiffies of utime+stime over wall time ÷ clock ticks), so the first <see cref="Sample"/> primes
    /// the baseline and returns 0 cores. All best-effort: returns 0 on any read/parse hiccup.
    /// </summary>
    public sealed class CpuRamSampler
    {
        private const double ClockTicksPerSecond = 100.0; // sysconf(_SC_CLK_TCK) — 100 on Android/Linux
        private long _lastJiffies;
        private long _lastMs;

        public (double coresBusy, long rssMb) Sample()
        {
            var nowMs = Android.OS.SystemClock.ElapsedRealtime();
            var jiffies = ReadProcessJiffies();

            var cores = 0.0;
            if (_lastMs > 0)
            {
                var dj = jiffies - _lastJiffies;
                var dt = (nowMs - _lastMs) / 1000.0;
                if (dt > 0)
                {
                    cores = (dj / ClockTicksPerSecond) / dt;
                }
            }

            _lastJiffies = jiffies;
            _lastMs = nowMs;
            return (cores, ReadRssMb());
        }

        // utime + stime (fields 14 & 15 of /proc/self/stat), in clock ticks. comm (field 2) may contain spaces
        // and parens, so fields are read AFTER the last ')'.
        private long ReadProcessJiffies()
        {
            try
            {
                var stat = System.IO.File.ReadAllText("/proc/self/stat");
                var afterComm = stat.LastIndexOf(')');
                if (afterComm < 0 || afterComm + 2 >= stat.Length)
                {
                    return _lastJiffies;
                }

                var fields = stat
                    .Substring(afterComm + 2)
                    .Split(' ', StringSplitOptions.RemoveEmptyEntries);
                // fields[0] = state (field 3); utime = field 14 → index 11; stime = field 15 → index 12.
                if (fields.Length > 12
                    && long.TryParse(fields[11], out var utime)
                    && long.TryParse(fields[12], out var stime))
                {
                    return utime + stime;
                }
            }
            catch
            {
                // best-effort
            }
            return _lastJiffies;
        }

        private static long ReadRssMb()
        {
            try
            {
                foreach (var line in System.IO.File.ReadLines("/proc/self/status"))
                {
                    if (!line.StartsWith("VmRSS:", StringComparison.Ordinal))
                    {
                        continue;
                    }
                    long kb = 0;
                    foreach (var ch in line)
                    {
                        if (ch >= '0' && ch <= '9')
                        {
                            kb = (kb * 10) + (ch - '0');
                        }
                    }
                    return kb / 1024;
                }
            }
            catch
            {
                // best-effort
            }
            return 0;
        }
    }
}
