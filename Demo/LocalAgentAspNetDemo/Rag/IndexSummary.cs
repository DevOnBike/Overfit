namespace DevOnBike.Overfit.Demo.LocalAgent.Rag
{
    public record IndexSummary(int TotalChunks, IReadOnlyList<FileIndexInfo> Files);
}