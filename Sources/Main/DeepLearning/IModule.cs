using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public interface IModule : IDisposable
    {
        bool IsTraining { get; }

        void Train();
        void Eval();

        // Główne przejście w przód
        AutogradNode Forward(ComputationGraph graph, AutogradNode input);

        // Pobieranie wag do Optymalizatora
        IEnumerable<AutogradNode> Parameters();

        // Zapis i odczyt
        void Save(BinaryWriter bw);
        void Load(BinaryReader br);
    }
}