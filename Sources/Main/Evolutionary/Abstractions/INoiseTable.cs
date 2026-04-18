namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    public interface INoiseTable
    {
        int Length { get; }
        ReadOnlySpan<float> GetSlice(int offset, int length);
    }
}