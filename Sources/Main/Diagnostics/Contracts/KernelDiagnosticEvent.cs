namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly record struct KernelDiagnosticEvent(
        string Category,
        string Name,
        double DurationMs,
        string Phase = "forward",
        bool IsTraining = false,
        int BatchSize = 0,
        int FeatureCount = 0,
        int InputElements = 0,
        int OutputElements = 0);
}
