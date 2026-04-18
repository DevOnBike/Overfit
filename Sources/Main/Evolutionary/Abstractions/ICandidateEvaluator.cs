namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    public interface ICandidateEvaluator<TContext>
    {
        float Evaluate(ReadOnlySpan<float> parameters, ref TContext context);
    }
}