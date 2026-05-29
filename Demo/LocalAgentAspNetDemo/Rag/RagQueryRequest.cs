namespace DevOnBike.Overfit.Demo.LocalAgent.Rag
{
    public record RagQueryRequest(string Question, int? TopK);
}