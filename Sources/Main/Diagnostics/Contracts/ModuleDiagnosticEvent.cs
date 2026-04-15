namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly record struct ModuleDiagnosticEvent(
        string ModuleType,
        string Phase,
        double DurationMs,
        int InputRows,
        int InputCols,
        long AllocatedBytes,
        bool IsTraining = false,
        int BatchSize = 0,
        int OutputRows = 0,
        int OutputCols = 0);
}
