using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class ReluActivation : IModule
    {
        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(AutogradNode input)
        {
            return TensorMath.ReLU(input);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            return Enumerable.Empty<AutogradNode>();
        }

        public void Save(BinaryWriter bw) { }
        public void Load(BinaryReader br) { }

        public void Dispose() { }
    }
}