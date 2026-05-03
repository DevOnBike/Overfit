namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    [Flags]
    public enum LoRATargetModules
    {
        None = 0,
        Query = 1 << 0,
        Key = 1 << 1,
        Value = 1 << 2,
        OutputProjection = 1 << 3,
        FeedForwardUp = 1 << 4,
        FeedForwardDown = 1 << 5,
        LanguageModelHead = 1 << 6,

        Attention = Query | Key | Value | OutputProjection,
        FeedForward = FeedForwardUp | FeedForwardDown,
        AllLinear = Attention | FeedForward | LanguageModelHead
    }
}