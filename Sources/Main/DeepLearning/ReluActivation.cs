using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class ReluActivation : IModule
    {
        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.ReLU(graph, input);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            return [];
        }

        public void Save(BinaryWriter bw) { }
        public void Load(BinaryReader br) { }

        public void Dispose() { }
    }
}