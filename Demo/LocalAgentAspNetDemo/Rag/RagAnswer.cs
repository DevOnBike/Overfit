namespace DevOnBike.Overfit.Demo.LocalAgent.Rag
{
    public record RagAnswer(
        string Reply,
        IReadOnlyList<RagSource> Sources,
        int PromptTokens,
        int GeneratedTokens,
        double TokensPerSecond,
        double SearchSeconds);
}