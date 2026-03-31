using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public interface IModule : IDisposable
    {
        bool IsTraining { get; }
        
        void Train(); 
        void Eval();
        
        // 1. Główne przejście w przód
        AutogradNode Forward(AutogradNode input);

        // 2. Pobieranie wag do Optymalizatora
        IEnumerable<AutogradNode> Parameters();

        // 3. Zapis i odczyt (Beast Mode)
        void Save(BinaryWriter bw);
        void Load(BinaryReader br);
    }

}