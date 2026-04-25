namespace DevOnBike.Overfit.Runtime
{
    public interface IRangeJob
    {
        void Execute(int startInclusive, int endExclusive);
    }
}