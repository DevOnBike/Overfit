using System.Buffers;
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

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            var maxHiddenSize = 65536;

            var bufA_Arr = ArrayPool<float>.Shared.Rent(maxHiddenSize);
            var bufB_Arr = ArrayPool<float>.Shared.Rent(maxHiddenSize);
            
            try
            {
                var bufA = bufA_Arr.AsSpan();
                var bufB = bufB_Arr.AsSpan();

                var currentInput = input;
                var currentOutput = bufA;

                // UWAGA: Zmień "_modules" na nazwę swojej tablicy/listy warstw w klasie Sequential.
                // Czasami nazywa się to Modules, _layers, albo children.
                var modulesList = _modules;

                // Jeśli modulesList to Lista (List<IModule>), użyj modulesList.Count zamiast modulesList.Length
                for (var i = 0; i < modulesList.Count; i++)
                {
                    // Ostatnia warstwa zapisuje bezpośrednio do docelowego 'output'
                    if (i == modulesList.Count - 1)
                    {
                        currentOutput = output;
                    }

                    modulesList[i].ForwardInference(currentInput, currentOutput);

                    // Ping-Pong: obecny output staje się inputem dla kolejnej warstwy
                    currentInput = currentOutput;
                    currentOutput = (currentOutput == bufA) ? bufB : bufA;
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(bufA_Arr);
                ArrayPool<float>.Shared.Return(bufB_Arr);
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