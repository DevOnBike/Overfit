namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly record struct ModuleDiagnosticEvent(
        string ModuleType,
        string Phase,
        double DurationMs,
        int InputRows,
        int InputCols,
        long AllocatedBytes);
}