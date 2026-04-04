namespace DevOnBike.Overfit.Data.Contracts
{
    public class ColumnDefinition
    {
        public string Name { get; set; }
        public ColumnType Type { get; set; }
    }
    
    internal struct FastTree
    {
        public int[] FeatureIndices;
        public float[] Thresholds;
        public float[] Values; // Średnia wartość w liściu (dla regresji ceny)
        public int Depth;
    }

}
