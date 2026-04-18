namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    public interface IBehaviorDescriptorEvaluator<TContext>
    {
        float Evaluate(ReadOnlySpan<float> parameters, ref TContext context, Span<float> descriptor);
    }
}