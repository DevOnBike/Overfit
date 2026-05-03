namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public interface ISlmModel : IDisposable
    {
        int VocabularySize { get; }

        int ContextLength { get; }

        int LayerCount { get; }

        int HeadCount { get; }

        int HeadDimension { get; }

        int EmbeddingDimension { get; }

        long ParameterCount { get; }

        string ArchitectureName { get; }

        bool IsTraining { get; }

        void Train();

        void Eval();
    }
}
