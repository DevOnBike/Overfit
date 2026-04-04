namespace DevOnBike.Overfit.Data.Contracts
{
    public class LayerDiagnostic
    {
        public string LayerName { get; init; }
        public int RowsBefore { get; init; }
        public int ColsBefore { get; init; }
        public int RowsAfter { get; init; }
        public int ColsAfter { get; init; }
        public long ElapsedMs { get; init; }
    }
}