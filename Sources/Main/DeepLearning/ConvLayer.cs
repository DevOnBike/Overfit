// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.DeepLearning.Diagnostics;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class ConvLayer : IModule
    {
        public AutogradNode Kernels { get; }
        public bool IsTraining { get; private set; } = true;

        private readonly int _inC;
        private readonly int _outC;
        private readonly int _h;
        private readonly int _w;
        private readonly int _k;

        public ConvLayer(int inChannels, int outChannels, int h, int w, int kSize)
        {
            _inC = inChannels;
            _outC = outChannels;
            _h = h;
            _w = w;
            _k = kSize;

            var kData = new TensorStorage<float>(outChannels * inChannels * kSize * kSize, clearMemory: false);
            InitializeKernels(kData.AsSpan(), inChannels * kSize * kSize);

            Kernels = new AutogradNode(kData, new TensorShape(outChannels, inChannels * kSize * kSize), requiresGrad: true);
        }

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var ctx = ModuleDiagnostics.Begin(nameof(ConvLayer), "forward_train", true, input.Shape.D0, _inC * _h * _w, input.Shape.D0, _outC * (_h - _k + 1) * (_w - _k + 1));
            try
            {
                return TensorMath.Conv2D(graph, input, Kernels, _inC, _outC, _h, _w, _k);
            }
            finally
            {
                ModuleDiagnostics.End(ctx);
            }
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Kernels;
        }

        public void InvalidateParameterCaches() { }

        public void Save(BinaryWriter bw)
        {
            bw.Write(Kernels.Shape.D0);
            bw.Write(Kernels.Shape.D1);

            foreach (var val in Kernels.DataView.AsReadOnlySpan())
            {
                bw.Write(val);
            }
        }

        public void Load(BinaryReader br)
        {
            var rows = br.ReadInt32();
            var cols = br.ReadInt32();

            if (rows != Kernels.Shape.D0 || cols != Kernels.Shape.D1)
            {
                throw new Exception("Kernel dimensions in file do not match the ConvLayer architecture.");
            }

            var span = Kernels.DataView.AsSpan();
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = br.ReadSingle();
            }
        }

        public void Dispose() => Kernels?.Dispose();

        private void InitializeKernels(Span<float> span, int fanIn)
        {
            var stdDev = MathF.Sqrt(2f / fanIn);
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = MathUtils.NextGaussian() * stdDev;
            }
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

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            throw new NotImplementedException("Bezalokacyjna inferencja SIMD dla Conv2D zostanie dodana w przyszłości.");
        }
    }
}