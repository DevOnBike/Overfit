namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    public interface ILoRAInjectable
    {
        bool SupportsLoRA { get; }

        ILoRAAdapter InjectLoRA(string name, in LoRAOptions options);

        bool TryGetLoRA(string name, out ILoRAAdapter? adapter);

        bool RemoveLoRA(string name);

        void FreezeBaseWeights();

        void UnfreezeBaseWeights();
    }
}
