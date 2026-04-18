namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    public interface IEliteArchive : IDisposable
    {
        int DescriptorDimensions { get; }

        bool TryInsert(ReadOnlySpan<float> parameters, float fitness, ReadOnlySpan<float> descriptor);
    }
}