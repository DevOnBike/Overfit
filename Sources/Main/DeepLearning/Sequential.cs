// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class Sequential : IModule
    {
        private readonly List<IModule> _modules = [];

        public Sequential(params IModule[] modules)
        {
            _modules.AddRange(modules);
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
                throw new FileNotFoundException($"Brak pliku filtr�w: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);

            Load(br);
        }
    }
}