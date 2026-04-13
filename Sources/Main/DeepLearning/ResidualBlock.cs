// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    ///     Implements a Residual Block (ResNet) to mitigate the vanishing gradient problem.
    /// </summary>
    public sealed class ResidualBlock : IModule
    {
        private readonly BatchNorm1D _bn1;
        private readonly BatchNorm1D _bn2;
        private readonly LinearLayer _linear1;
        private readonly LinearLayer _linear2;

        public ResidualBlock(int hiddenSize)
        {
            _linear1 = new LinearLayer(hiddenSize, hiddenSize);
            _bn1 = new BatchNorm1D(hiddenSize);
            _linear2 = new LinearLayer(hiddenSize, hiddenSize);
            _bn2 = new BatchNorm1D(hiddenSize);
        }

        public bool IsTraining { get; private set; } = true;

        public void Train()
        {
            IsTraining = true;
            _linear1.Train(); _bn1.Train(); _linear2.Train(); _bn2.Train();
        }

        public void Eval()
        {
            IsTraining = false;
            _linear1.Eval(); _bn1.Eval(); _linear2.Eval(); _bn2.Eval();
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            if (graph == null || !IsTraining)
            {
                var out1 = _linear1.Forward(null, input);
                using var bn1Out = _bn1.Forward(null, out1);
                using var a1 = TensorMath.ReLU(null, bn1Out);

                var out2 = _linear2.Forward(null, a1);
                using var bn2Out = _bn2.Forward(null, out2);

                using var added = TensorMath.Add(null, bn2Out, input);

                return TensorMath.ReLU(null, added);
            }

            var tOut1 = _linear1.Forward(graph, input);
            var tBn1 = _bn1.Forward(graph, tOut1);
            var tA1 = TensorMath.ReLU(graph, tBn1);

            var tOut2 = _linear2.Forward(graph, tA1);
            var tBn2 = _bn2.Forward(graph, tOut2);

            var tAdded = TensorMath.Add(graph, tBn2, input);

            return TensorMath.ReLU(graph, tAdded);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            foreach (var p in _linear1.Parameters())
            {
                yield return p;
            }
            foreach (var p in _bn1.Parameters())
            {
                yield return p;
            }
            foreach (var p in _linear2.Parameters())
            {
                yield return p;
            }
            foreach (var p in _bn2.Parameters())
            {
                yield return p;
            }
        }

        public void Save(BinaryWriter bw)
        {
            _linear1.Save(bw); _bn1.Save(bw); _linear2.Save(bw); _bn2.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            _linear1.Load(br); _bn1.Load(br); _linear2.Load(br); _bn2.Load(br);
        }

        public void Dispose()
        {
            _linear1?.Dispose(); _bn1?.Dispose(); _linear2?.Dispose(); _bn2?.Dispose();
        }

        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);
            Save(bw);
        }

        public void Load(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Model weights file not found: {path}");
            }
            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);
            Load(br);
        }
    }
}