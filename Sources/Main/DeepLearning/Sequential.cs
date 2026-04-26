// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Licensing;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class Sequential : IModule
    {
        private const int DefaultInferenceBufferSize = 65_536;

        private readonly List<IModule> _modules = [];

        private float[] _inferenceBufferA = [];
        private float[] _inferenceBufferB = [];

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

                if (module is IInferenceShapeProvider inferenceModule)
                {
                    inferenceModule.PrepareInference();
                }
            }

            EnsureInferenceCapacity(DefaultInferenceBufferSize);
        }

        /// <summary>
        /// Allocates reusable inference workspace outside the inference hot path.
        /// Call this before BenchmarkDotNet measurement when the model has hidden
        /// tensors larger than the default buffer.
        /// </summary>
        public void PrepareInference(int maxIntermediateElements = DefaultInferenceBufferSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(maxIntermediateElements);

            foreach (var module in _modules)
            {
                if (module is IInferenceShapeProvider inferenceModule)
                {
                    inferenceModule.PrepareInference();
                }
            }

            EnsureInferenceCapacity(maxIntermediateElements);
        }

        private void EnsureInferenceCapacity(int requiredElements)
        {
            if (_inferenceBufferA.Length < requiredElements)
            {
                _inferenceBufferA = new float[requiredElements];
            }

            if (_inferenceBufferB.Length < requiredElements)
            {
                _inferenceBufferB = new float[requiredElements];
            }
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            OverfitLicense.EnsureNotified();

            var count = _modules.Count;

            if (count == 0)
            {
                input.CopyTo(output);
                return;
            }

            if (count == 1)
            {
                _modules[0].ForwardInference(input, output);
                return;
            }

            EnsureInferenceCapacity(DefaultInferenceBufferSize);

            ReadOnlySpan<float> currentInput = input;
            var currentInputLength = input.Length;

            var useA = true;

            for (var i = 0; i < count; i++)
            {
                var isLast = i == count - 1;
                var module = _modules[i];

                var currentOutputLength = isLast
                    ? output.Length
                    : GetInferenceOutputLength(module, currentInputLength);

                Span<float> currentOutput;

                if (isLast)
                {
                    currentOutput = output;
                }
                else
                {
                    if (currentOutputLength > _inferenceBufferA.Length)
                    {
                        // This allocation is deliberately not hidden.
                        // Call PrepareInference(maxIntermediateElements) before hot-path inference.
                        EnsureInferenceCapacity(currentOutputLength);
                    }

                    currentOutput = useA
                        ? _inferenceBufferA.AsSpan(0, currentOutputLength)
                        : _inferenceBufferB.AsSpan(0, currentOutputLength);
                }

                module.ForwardInference(
                    currentInput.Slice(0, currentInputLength),
                    currentOutput);

                currentInput = currentOutput;
                currentInputLength = currentOutputLength;
                useA = !useA;
            }
        }

        private static int GetInferenceOutputLength(IModule module, int inputLength)
        {
            if (module is IInferenceShapeProvider shapeProvider)
            {
                var inputSize = shapeProvider.InferenceInputSize;

                if (inputSize <= 0 || inputLength % inputSize != 0)
                {
                    throw new InvalidOperationException(
                        $"Cannot infer output length for module {module.GetType().Name}.");
                }

                var batchSize = inputLength / inputSize;
                return batchSize * shapeProvider.InferenceOutputSize;
            }

            // ReLU/Tanh and other element-wise activations keep the same length.
            return inputLength;
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            OverfitLicense.EnsureNotified();

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

            InvalidateParameterCaches();
        }

        public void InvalidateParameterCaches()
        {
            foreach (var module in _modules)
            {
                module.InvalidateParameterCaches();

                if (!IsTraining && module is IInferenceShapeProvider inferenceModule)
                {
                    inferenceModule.PrepareInference();
                }
            }
        }

        public void Dispose()
        {
            foreach (var module in _modules)
            {
                module.Dispose();
            }

            _modules.Clear();
            _inferenceBufferA = [];
            _inferenceBufferB = [];
        }

        public void Add(IModule module)
        {
            _modules.Add(module);
            InvalidateParameterCaches();
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