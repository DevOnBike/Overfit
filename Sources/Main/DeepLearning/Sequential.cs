// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class Sequential : IModule
    {
        private readonly List<IModule> _modules = [];

        public Sequential(params ReadOnlySpan<IModule> modules)
        {
            foreach (var module in modules)
            {
                _modules.Add(module);
            }
        }

        public bool IsTraining { get; private set; } = true;

        public void Train()
        {
            IsTraining = true;

            foreach (var module in _modules)
            {
                module.Train();
            }
        }

        public void Eval()
        {
            IsTraining = false;

            foreach (var module in _modules)
            {
                module.Eval();
            }
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            var maxHiddenSize = 65536;

            using var bufA_Buf = new PooledBuffer<float>(maxHiddenSize);
            using var bufB_Buf = new PooledBuffer<float>(maxHiddenSize);

            var bufA = bufA_Buf.Span;
            var bufB = bufB_Buf.Span;

            var currentInput = input;
            var currentOutput = bufA;

            var modulesList = _modules;

            for (var i = 0; i < modulesList.Count; i++)
            {
                if (i == modulesList.Count - 1)
                {
                    currentOutput = output;
                }

                modulesList[i].ForwardInference(currentInput, currentOutput);

                currentInput = currentOutput;
                currentOutput = (currentOutput == bufA) ? bufB : bufA;
            }
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var current = input;

            foreach (var module in _modules)
            {
                current = module.Forward(graph, current);
            }

            return current;
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            foreach (var module in _modules)
            {
                foreach (var p in module.Parameters())
                {
                    yield return p;
                }
            }
        }

        public void Save(BinaryWriter bw)
        {
            foreach (var module in _modules)
            {
                module.Save(bw);
            }
        }

        public void Load(BinaryReader br)
        {
            foreach (var module in _modules)
            {
                module.Load(br);
            }
        }

        public void Dispose()
        {
            foreach (var module in _modules)
            {
                module.Dispose();
            }

            _modules.Clear();
        }

        public void Add(IModule module)
        {
            _modules.Add(module);
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
                throw new FileNotFoundException($"Brak pliku modelu: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);

            Load(br);
        }
    }
}