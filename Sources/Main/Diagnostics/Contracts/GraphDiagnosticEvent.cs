namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly record struct GraphDiagnosticEvent(
        int TapeOpCount,
        double BackwardMs,
        long AllocatedBytes,
        int Gen0Collections,
        int Gen1Collections,
        int Gen2Collections);
}