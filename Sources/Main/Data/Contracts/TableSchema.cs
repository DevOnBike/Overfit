namespace DevOnBike.Overfit.Data.Contracts
{
    public class TableSchema
    {
        public List<ColumnDefinition> Features { get; set; } = [];
        public ColumnDefinition Target { get; set; }
    }
}