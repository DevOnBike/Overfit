namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    public interface IParameterVectorAdapter
    {
        int ParameterCount { get; }

        void WriteToVector(Span<float> destination);
        void ReadFromVector(ReadOnlySpan<float> source);
    }
}