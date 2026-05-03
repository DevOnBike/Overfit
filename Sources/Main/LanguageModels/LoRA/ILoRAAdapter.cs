namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    public interface ILoRAAdapter : IDisposable
    {
        string Name { get; }

        LoRAOptions Options { get; }

        long TrainableParameterCount { get; }

        bool IsEnabled { get; }

        void Enable();

        void Disable();

        void Save(string path);

        void Load(string path);

        void ZeroGrad();
    }
}
