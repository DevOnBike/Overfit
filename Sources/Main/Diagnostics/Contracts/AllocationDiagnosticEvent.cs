namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly record struct AllocationDiagnosticEvent(
        string Owner,
        string ResourceType,
        int Elements,
        long Bytes,
        bool IsPooled = false,
        bool IsManaged = true);
}