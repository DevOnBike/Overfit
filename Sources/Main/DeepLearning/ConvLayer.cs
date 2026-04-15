using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.DeepLearning.Diagnostics;
using DevOnBike.Overfit.Diagnostics.Contracts;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;

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

            var kData = new FastTensor<float>(outChannels, inChannels * kSize * kSize, clearMemory: false);

            InitializeKernels(kData.GetView().AsSpan(), inChannels * kSize * kSize);

            Kernels = new AutogradNode(kData);
        }

        public void Train()
        {
            IsTraining = true;
        }

        public void Eval()
        {
            IsTraining = false;
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            var outH = _h - _k + 1;
            var outW = _w - _k + 1;
            var kSqInC = _k * _k * _inC;
            var outElements = outH * outW;

            using var colArr = new PooledBuffer<float>(kSqInC * outElements);
            
            var colSpan = colArr.Span;

            using (new DiagnosticScope(
                   category: "DeepLearning",
                   name: "ConvLayer.ForwardInference",
                   phase: "forward_inference",
                   isTraining: false,
                   batchSize: 1,
                   featureCount: _outC,
                   inputElements: input.Length,
                   outputElements: output.Length))
            {
                TensorMath.Im2Col(input, _inC, _h, _w, _k, 1, 0, colSpan);
                var w2D = Kernels.DataView.AsReadOnlySpan();
                TensorMath.MatMulRawSeq(w2D, colSpan, _outC, kSqInC, outElements, output);
            }
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var batch = input.DataView.GetDim(0);
            var outH = _h - _k + 1;
            var outW = _w - _k + 1;

            var ctx = ModuleDiagnostics.Begin(
            moduleType: nameof(ConvLayer),
            phase: graph == null || !IsTraining ? "forward_eval" : "forward_train",
            isTraining: IsTraining,
            batchSize: batch,
            inputRows: batch,
            inputCols: _inC * _h * _w,
            outputRows: batch,
            outputCols: _outC * outH * outW);

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

        public void Save(BinaryWriter bw)
        {
            bw.Write(Kernels.DataView.GetDim(0));
            bw.Write(Kernels.DataView.GetDim(1));

            foreach (var val in Kernels.DataView.AsSpan())
            {
                bw.Write(val);
            }
        }

        public void Load(BinaryReader br)
        {
            var rows = br.ReadInt32();
            var cols = br.ReadInt32();

            if (rows != Kernels.DataView.GetDim(0) || cols != Kernels.DataView.GetDim(1))
            {
                throw new Exception("Kernel dimensions in file do not match the ConvLayer architecture.");
            }

            var span = Kernels.DataView.AsSpan();
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = br.ReadSingle();
            }
        }

        public void Dispose()
        {
            Kernels?.Dispose();
        }

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
    }
}
