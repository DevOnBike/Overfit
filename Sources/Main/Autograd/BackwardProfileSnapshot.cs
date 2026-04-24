using System.Diagnostics;

namespace DevOnBike.Overfit.Autograd
{
    public sealed class BackwardProfileSnapshot
    {
        private readonly BackwardOpProfile[] _profiles;

        public BackwardProfileSnapshot(
            BackwardOpProfile[] profiles,
            long totalElapsedTicks,
            long totalAllocatedBytes)
        {
            _profiles = profiles ?? Array.Empty<BackwardOpProfile>();
            TotalElapsedTicks = totalElapsedTicks;
            TotalAllocatedBytes = totalAllocatedBytes;
        }

        public long TotalElapsedTicks { get; }
        public long TotalAllocatedBytes { get; }

        public double TotalElapsedMs => TotalElapsedTicks * 1000.0 / Stopwatch.Frequency;
        public double TotalAllocatedMb => TotalAllocatedBytes / 1024.0 / 1024.0;

        public ReadOnlySpan<BackwardOpProfile> Profiles => _profiles;

        public bool IsEmpty => _profiles.Length == 0;
    }
}