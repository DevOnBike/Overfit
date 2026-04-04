using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class Sequential : IModule
    {
        private readonly List<IModule> _modules = [];

        public bool IsTraining { get; private set; } = true;

        public Sequential(params IModule[] modules)
        {
            _modules.AddRange(modules);
        }

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

        public void Add(IModule module)
        {
            _modules.Add(module);
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
            return _modules.SelectMany(m => m.Parameters());
        }

        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);

            Save(bw);
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
    }
}