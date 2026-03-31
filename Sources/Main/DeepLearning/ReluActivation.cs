using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class ReluActivation : IModule // Sealed jak inne Twoje warstwy[cite: 4, 6]
    {
        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(AutogradNode input)
        {
            // Korzystamy z metody statycznej w TensorMath, którą widać w Twoim ResidualBlock[cite: 7]
            return TensorMath.ReLU(input);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            // ReLU nie ma parametrów (wag), więc zwracamy pustą listę
            return Enumerable.Empty<AutogradNode>();
        }

        // Metody zapisu/odczytu są puste, bo ReLU nie ma stanu do zachowania
        public void Save(BinaryWriter bw) { }
        public void Load(BinaryReader br) { }

        public void Dispose() { }
    }
}