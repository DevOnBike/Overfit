using System.Runtime.InteropServices;
namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public readonly record struct RawSample
    {
        public long Timestamp { get; init; }
        public float Value { get; init; }
    }
}