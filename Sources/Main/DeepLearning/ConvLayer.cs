// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Implements a 2D Convolutional Layer for spatial feature extraction.
    /// Uses an optimized weight format [outChannels, inChannels * kSize * kSize] 
    /// to leverage high-speed matrix multiplication (Im2Col).
    /// </summary>
    public sealed class ConvLayer : IModule
    {
        /// <summary> The learnable kernels (filters) of the layer. </summary>
        public AutogradNode Kernels { get; }

        private readonly int _inC, _outC, _h, _w, _k;

        public bool IsTraining { get; private set; } = true;

        /// <param name="inChannels">Number of input channels (e.g., 3 for RGB).</param>
        /// <param name="outChannels">Number of filters (kernels) to apply.</param>
        /// <param name="h">Input height.</param>
        /// <param name="w">Input width.</param>
        /// <param name="kSize">Kernel (filter) size (e.g., 3 for a 3x3 filter).</param>
        public ConvLayer(int inChannels, int outChannels, int h, int w, int kSize)
        {
            _inC = inChannels;
            _outC = outChannels;
            _h = h;
            _w = w;
            _k = kSize;

            // Weights are stored in a flattened 2D format to facilitate Im2Col optimization
            var kData = new FastTensor<float>(outChannels, inChannels * kSize * kSize);
            InitializeKernels(kData.AsSpan(), inChannels * kSize * kSize);

            Kernels = new AutogradNode(kData, true);
        }

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        /// <summary>
        /// Initializes kernels using He (Kaiming) initialization: stdDev = sqrt(2 / fanIn).
        /// Ideal for layers followed by ReLU activation.
        /// </summary>
        private void InitializeKernels(Span<float> span, int fanIn)
        {
            var stdDev = MathF.Sqrt(2f / fanIn);

            for (var i = 0; i < span.Length; i++)
            {
                span[i] = MathUtils.NextGaussian() * stdDev;
            }
        }

        /// <summary>
        /// Performs the 2D convolution forward pass via the global math engine.
        /// </summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.Conv2D(graph, input, Kernels, _inC, _outC, _h, _w, _k);
        }

        public IEnumerable<AutogradNode> Parameters() { yield return Kernels; }

        /// <summary> Saves kernels dimensions and raw data to a binary stream. </summary>
        public void Save(BinaryWriter bw)
        {
            bw.Write(Kernels.Data.Shape[0]);
            bw.Write(Kernels.Data.Shape[1]);
            foreach (var val in Kernels.Data.AsSpan()) bw.Write(val);
        }

        /// <summary> 
        /// Loads kernels from a binary stream. 
        /// Performs shape validation to prevent loading incompatible weights.
        /// </summary>
        public void Load(BinaryReader br)
        {
            var rows = br.ReadInt32();
            var cols = br.ReadInt32();

            if (rows != Kernels.Data.Shape[0] || cols != Kernels.Data.Shape[1])
            {
                throw new Exception("Kernel dimensions in file do not match the ConvLayer architecture.");
            }

            var span = Kernels.Data.AsSpan();
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = br.ReadSingle();
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

        public void Dispose() => Kernels?.Dispose();
    }
}