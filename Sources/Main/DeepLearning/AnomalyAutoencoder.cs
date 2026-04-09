// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using System.Linq;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// MLP Autoencoder for K8s metrics anomaly detection.
    ///
    /// Architecture (default, inputSize=32):
    ///   Encoder: 32 → 16 → 8 → 4  (bottleneck)
    ///   Decoder:  4 →  8 → 16 → 32
    ///   Every layer except the last one: LinearLayer → BatchNorm1D → ReLU
    ///   Last decoder layer: LinearLayer without activation (reconstruction of normalized values)
    ///
    /// Input:  output of FeatureExtractor after RobustScaler — flat float[inputSize]
    /// Output: reconstruction float[inputSize]
    ///
    /// Inference: <see cref="Reconstruct"/> — zero-allocation SIMD path through LinearLayer.
    ///   Call <see cref="Eval"/> before production use.
    ///
    /// Training:
    /// <code>
    ///   autoencoder.Train();
    ///   using var graph  = new ComputationGraph();
    ///   using var input  = new AutogradNode(new FastTensor&lt;float&gt;(1, inputSize), false);
    ///   features.CopyTo(input.Data.AsSpan());
    ///   var recon = autoencoder.Forward(graph, input);
    ///   var loss  = TensorMath.MSELoss(graph, recon, input);
    ///   loss.Backward();
    ///   optimizer.Step();
    /// </code>
    /// </summary>
    public sealed class AnomalyAutoencoder : IModule
    {
        private readonly Sequential _encoder;
        private readonly Sequential _decoder;

        // Preallocated input node [1, InputSize] — reused on every Reconstruct call.
        // Shape [1, x] activates the zero-allocation SIMD path in LinearLayer (batchSize=1).
        private readonly AutogradNode _inputNode;
        private bool _disposed;

        public int InputSize { get; }
        public int BottleneckDim { get; }
        public bool IsTraining { get; private set; } = true;

        // -------------------------------------------------------------------------
        // Constructor
        // -------------------------------------------------------------------------

        /// <param name="inputSize">
        ///   Input vector size = FeatureExtractor.OutputSize(featureCount).
        ///   Default is 32 (8 features × 4 statistics).
        /// </param>
        /// <param name="hidden1">First hidden layer size. Default is inputSize/2.</param>
        /// <param name="hidden2">Second hidden layer size. Default is inputSize/4.</param>
        /// <param name="bottleneckDim">Bottleneck dimension (latent space). Default is inputSize/8.</param>
        public AnomalyAutoencoder(
            int inputSize,
            int hidden1 = 0, // 0 = inputSize / 2
            int hidden2 = 0, // 0 = inputSize / 4
            int bottleneckDim = 0) // 0 = inputSize / 8
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);

            InputSize = inputSize;
            hidden1 = hidden1 > 0 ? hidden1 : Math.Max(1, inputSize / 2);
            hidden2 = hidden2 > 0 ? hidden2 : Math.Max(1, inputSize / 4);
            bottleneckDim = bottleneckDim > 0 ? bottleneckDim : Math.Max(1, inputSize / 8);

            BottleneckDim = bottleneckDim;

            _encoder = new Sequential(
                new LinearLayer(inputSize, hidden1),
                new BatchNorm1D(hidden1),
                new ReluActivation(),

                new LinearLayer(hidden1, hidden2),
                new BatchNorm1D(hidden2),
                new ReluActivation(),

                new LinearLayer(hidden2, bottleneckDim),
                new BatchNorm1D(bottleneckDim),
                new ReluActivation()
            );

            _decoder = new Sequential(
                new LinearLayer(bottleneckDim, hidden2),
                new BatchNorm1D(hidden2),
                new ReluActivation(),

                new LinearLayer(hidden2, hidden1),
                new BatchNorm1D(hidden1),
                new ReluActivation(),

                // Last layer without activation — reconstructs normalized values ∈ (-∞, +∞)
                new LinearLayer(hidden1, inputSize)
            );

            // [1, InputSize] — shape required by the SIMD inference path in LinearLayer
            _inputNode = new AutogradNode(new FastTensor<float>(1, inputSize), requiresGrad: false);
        }

        // -------------------------------------------------------------------------
        // Train / Eval
        // -------------------------------------------------------------------------

        public void Train()
        {
            IsTraining = true;

            _encoder.Train();
            _decoder.Train();
        }

        public void Eval()
        {
            IsTraining = false;

            _encoder.Eval();
            _decoder.Eval();
        }

        // -------------------------------------------------------------------------
        // Inference — zero-allocation path
        // -------------------------------------------------------------------------

        /// <summary>
        /// Reconstructs the input through the autoencoder.
        /// Uses the preallocated _inputNode [1, InputSize] -> SIMD path in LinearLayer.
        ///
        /// IMPORTANT: call <see cref="Eval"/> before the first production use,
        /// so BatchNorm uses running statistics instead of batch statistics.
        /// </summary>
        /// <param name="features">Normalized features from RobustScaler. Size must be == InputSize.</param>
        /// <param name="reconstruction">Caller-owned result buffer. Size must be >= InputSize.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Reconstruct(ReadOnlySpan<float> features, Span<float> reconstruction)
        {
            if (features.Length != InputSize)
            {
                throw new ArgumentException($"Oczekiwano {InputSize} cech, otrzymano {features.Length}.", nameof(features));
            }

            if (reconstruction.Length < InputSize)
            {
                throw new ArgumentException($"Bufor rekonstrukcji za krótki: potrzeba {InputSize}, dostępne {reconstruction.Length}.", nameof(reconstruction));
            }

            // Copy to the preallocated tensor — one copy per call, no allocation
            features.CopyTo(_inputNode.Data.AsSpan());

            // graph=null -> zero-allocation SIMD path in all LinearLayers
            var latent = _encoder.Forward(null, _inputNode);
            var output = _decoder.Forward(null, latent);

            // Copy the result before the next Forward call (output node is reused by LinearLayer)
            output.Data.AsSpan().CopyTo(reconstruction);
        }

        // -------------------------------------------------------------------------
        // Training forward — full autograd
        // -------------------------------------------------------------------------

        /// <summary>
        /// Forward pass with optional recording to the computation graph.
        /// Use during training:
        ///   var recon = autoencoder.Forward(graph, inputNode);
        ///   var loss  = TensorMath.MSELoss(graph, recon, inputNode);
        /// </summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var latent = _encoder.Forward(graph, input);

            return _decoder.Forward(graph, latent);
        }

        // -------------------------------------------------------------------------
        // Parameters, serialization, diagnostics
        // -------------------------------------------------------------------------

        /// <summary>
        /// Returns all learnable parameters of the encoder and decoder.
        /// Pass to the optimizer: <c>new Adam(autoencoder.Parameters(), lr)</c>.
        /// </summary>
        public IEnumerable<AutogradNode> Parameters()
        {
            foreach (var p in _encoder.Parameters())
            {
                yield return p;
            }
            
            foreach (var p in _decoder.Parameters())
            {
                yield return p;
            }
        }

        /// <summary>Number of learnable parameters (weights + biases + gamma + beta of BN).</summary>
        public int ParameterCount => Parameters().Sum(p => p.Data.Size);

        public void Save(BinaryWriter bw)
        {
            _encoder.Save(bw);
            _decoder.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            _encoder.Load(br);
            _decoder.Load(br);
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

        public void Dispose()
        {
            if (_disposed) { return; }
            _disposed = true;
            _encoder.Dispose();
            _decoder.Dispose();
            _inputNode.Dispose();
        }
    }
}