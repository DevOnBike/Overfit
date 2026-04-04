using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class ResidualBlock : IModule
    {
        private readonly LinearLayer _linear1;
        private readonly BatchNorm1D _bn1;
        private readonly LinearLayer _linear2;
        private readonly BatchNorm1D _bn2;

        public bool IsTraining { get; private set; } = true;

        public ResidualBlock(int hiddenSize)
        {
            _linear1 = new LinearLayer(hiddenSize, hiddenSize);
            _bn1 = new BatchNorm1D(hiddenSize);
            _linear2 = new LinearLayer(hiddenSize, hiddenSize);
            _bn2 = new BatchNorm1D(hiddenSize);
        }

        public void Train()
        {
            IsTraining = true;
            _linear1.Train();
            _bn1.Train();
            _linear2.Train();
            _bn2.Train();
        }

        public void Eval()
        {
            IsTraining = false;
            _linear1.Eval();
            _bn1.Eval();
            _linear2.Eval();
            _bn2.Eval();
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var out1 = _linear1.Forward(graph, input);
            var bn1Out = _bn1.Forward(graph, out1);
            var a1 = TensorMath.ReLU(graph, bn1Out);

            var out2 = _linear2.Forward(graph, a1);
            var bn2Out = _bn2.Forward(graph, out2);

            var added = TensorMath.Add(graph, bn2Out, input);

            return TensorMath.ReLU(graph, added);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            return _linear1.Parameters()
                .Concat(_bn1.Parameters())
                .Concat(_linear2.Parameters())
                .Concat(_bn2.Parameters());
        }

        public void Save(BinaryWriter bw)
        {
            _linear1.Save(bw);
            _bn1.Save(bw);
            _linear2.Save(bw);
            _bn2.Save(bw);
        }

        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);

            Save(bw);
        }

        public void Load(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Brak pliku wag: {path}");
            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);

            Load(br);
        }

        public void Load(BinaryReader br)
        {
            _linear1.Load(br);
            _bn1.Load(br);
            _linear2.Load(br);
            _bn2.Load(br);
        }

        public void Dispose()
        {
            _linear1?.Dispose();
            _bn1?.Dispose();
            _linear2?.Dispose();
            _bn2?.Dispose();
        }
    }
}