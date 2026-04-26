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
        private const int DefaultInferenceBufferSize = 64 * 1024;

        private readonly List<IModule> _modules = [];

        private float[] _inferenceBufferA = [];
        private float[] _inferenceBufferB = [];

        private bool _inferencePrepared;
        private int _inferencePreparedCapacity;

        public Sequential(params IModule[] modules)
        {
            ArgumentNullException.ThrowIfNull(modules);

            foreach (var module in modules)
            {
                ArgumentNullException.ThrowIfNull(module);
                _modules.Add(module);
            }
        }

        public bool IsTraining { get; private set; } = true;

        public void Train()
        {
            IsTraining = true;
            _inferencePrepared = false;

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

            PrepareInference(DefaultInferenceBufferSize);
        }

        public void PrepareInference(
            int maxIntermediateElements = DefaultInferenceBufferSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxIntermediateElements);

            EnsureInferenceCapacitySlow(maxIntermediateElements);

            foreach (var module in _modules)
            {
                if (module is IInferenceShapeProvider inferenceShapeProvider)
                {
                    inferenceShapeProvider.PrepareInference();
                }
            }

            _inferencePreparedCapacity = maxIntermediateElements;
            _inferencePrepared = true;
        }

        public void ForwardInference(
            ReadOnlySpan<float> input,
            Span<float> output)
        {
            OverfitLicense.EnsureNotified();

            if (!_inferencePrepared)
            {
                PrepareInference(DefaultInferenceBufferSize);
            }

            ForwardInferencePrepared(input, output);
        }

        private void ForwardInferencePrepared(
            ReadOnlySpan<float> input,
            Span<float> output)
        {
            if (_modules.Count == 0)
            {
                throw new InvalidOperationException("Sequential contains no modules.");
            }

            if (_modules.Count == 1)
            {
                InvokePreparedOrSafe(
                    _modules[0],
                    input,
                    output);

                return;
            }

            var currentInput = input;
            var useBufferA = true;

            for (var i = 0; i < _modules.Count; i++)
            {
                var module = _modules[i];
                var isLast = i == _modules.Count - 1;

                if (isLast)
                {
                    InvokePreparedOrSafe(
                        module,
                        currentInput,
                        output);

                    return;
                }

                var nextLength = ResolveIntermediateLength(
                    module,
                    currentInput);

                EnsurePreparedCapacityForIntermediate(nextLength);

                var currentOutput = useBufferA
                    ? _inferenceBufferA.AsSpan(0, nextLength)
                    : _inferenceBufferB.AsSpan(0, nextLength);

                InvokePreparedOrSafe(
                    module,
                    currentInput,
                    currentOutput);

                currentInput = currentOutput;
                useBufferA = !useBufferA;
            }
        }

        private static void InvokePreparedOrSafe(
            IModule module,
            ReadOnlySpan<float> input,
            Span<float> output)
        {
            if (module is IPreparedInferenceModule preparedModule)
            {
                preparedModule.ForwardInferencePrepared(
                    input,
                    output);

                return;
            }

            module.ForwardInference(
                input,
                output);
        }

        private static int ResolveIntermediateLength(
            IModule module,
            ReadOnlySpan<float> currentInput)
        {
            if (module is not IInferenceShapeProvider shapeProvider)
            {
                // Shape-preserving modules, e.g. ReLU.
                return currentInput.Length;
            }

            if (currentInput.Length % shapeProvider.InferenceInputSize != 0)
            {
                throw new ArgumentException(
                    $"Input length for module {module.GetType().Name} is not divisible by " +
                    $"InferenceInputSize={shapeProvider.InferenceInputSize}.");
            }

            var batchSize = currentInput.Length / shapeProvider.InferenceInputSize;

            return batchSize * shapeProvider.InferenceOutputSize;
        }

        public AutogradNode Forward(
            ComputationGraph graph,
            AutogradNode input)
        {
            OverfitLicense.EnsureNotified();

            var current = input;

            foreach (var module in _modules)
            {
                current = module.Forward(
                    graph,
                    current);
            }

            return current;
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            foreach (var module in _modules)
            {
                foreach (var parameter in module.Parameters())
                {
                    yield return parameter;
                }
            }
        }

        public void Save(
            BinaryWriter bw)
        {
            foreach (var module in _modules)
            {
                module.Save(bw);
            }
        }

        public void Load(
            BinaryReader br)
        {
            foreach (var module in _modules)
            {
                module.Load(br);
            }

            _inferencePrepared = false;
        }

        public void InvalidateParameterCaches()
        {
            _inferencePrepared = false;

            foreach (var module in _modules)
            {
                module.InvalidateParameterCaches();
            }
        }

        public void Add(
            IModule module)
        {
            ArgumentNullException.ThrowIfNull(module);

            _modules.Add(module);
            _inferencePrepared = false;
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
            _inferencePrepared = false;
            _inferencePreparedCapacity = 0;
        }

        public void Save(
            string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);

            Save(bw);
        }

        public void Load(
            string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Brak pliku modelu: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);

            Load(br);
        }

        private void EnsureInferenceCapacitySlow(
            int requiredElements)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(requiredElements);

            if (_inferenceBufferA.Length < requiredElements)
            {
                _inferenceBufferA = new float[requiredElements];
            }

            if (_inferenceBufferB.Length < requiredElements)
            {
                _inferenceBufferB = new float[requiredElements];
            }
        }

        private void EnsurePreparedCapacityForIntermediate(
            int requiredElements)
        {
            if (requiredElements <= _inferencePreparedCapacity &&
                requiredElements <= _inferenceBufferA.Length &&
                requiredElements <= _inferenceBufferB.Length)
            {
                return;
            }

            PrepareInference(
                Math.Max(
                    requiredElements,
                    DefaultInferenceBufferSize));
        }
    }
}