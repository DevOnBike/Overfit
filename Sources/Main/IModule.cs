
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit
{
    public interface IModule
    {
        // Główna operacja. isTraining mówi, czy np. Dropout albo BatchNorm mają być aktywne
        AutogradNode Forward(AutogradNode input, bool isTraining = true);
    
        // Zwraca wagi i biasy, żeby Optymalizator i Janitor wiedzieli, czego nie kasować
        IEnumerable<AutogradNode> Parameters();
    
        // Zapis/Odczyt stanu
        void Save(BinaryWriter writer);
        void Load(BinaryReader reader);
    }

}