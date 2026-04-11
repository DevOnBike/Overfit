// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// LSTM Autoencoder for time-series anomaly detection.
    ///
    /// Architecture:
    ///   Encoder:
    ///     LSTMLayer(inputSize → encoderHidden, returnSequences:true)
    ///     LSTMLayer(encoderHidden → latentSize, returnSequences:false)  → latent [batch, latentSize]
    ///
    ///   Decoder:
    ///     RepeatVector(seqLen)                                          → [batch, seqLen, latentSize]
    ///     LSTMLayer(latentSize → decoderHidden, returnSequences:true)
    ///     LSTMLayer(decoderHidden → inputSize, returnSequences:true)    → [batch, seqLen, inputSize]
    ///
    /// Loss: MSE(input, reconstruction)
    ///
    /// Inference:
    ///   Call Eval() before use.
    ///   ReconstructionError(input) returns per-window MSE — threshold this for anomaly detection.
    /// </summary>
    public sealed class LSTMAutoencoder : IModule
    {
        private readonly LSTMLayer _enc1;
        private readonly LSTMLayer _enc2;
        private readonly RepeatVector _repeat;
        private readonly LSTMLayer _dec1;
        private readonly LSTMLayer _dec2;

        public bool IsTraining { get; private set; } = true;

        public int InputSize { get; }
        public int SeqLen { get; }
        public int EncoderHidden { get; }
        public int LatentSize { get; }
        public int DecoderHidden { get; }

        /// <param name="inputSize">Number of features per timestep (e.g. 12 metrics).</param>
        /// <param name="seqLen">Sequence length (e.g. 60 timesteps).</param>
        /// <param name="encoderHidden">Hidden size of first encoder LSTM layer.</param>
        /// <param name="latentSize">Bottleneck dimension — size of the latent vector.</param>
        /// <param name="decoderHidden">Hidden size of first decoder LSTM layer.</param>
        public LSTMAutoencoder(
            int inputSize,
            int seqLen,
            int encoderHidden = 64,
            int latentSize = 32,
            int decoderHidden = 64)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(encoderHidden);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(latentSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(decoderHidden);

            InputSize = inputSize;
            SeqLen = seqLen;
            EncoderHidden = encoderHidden;
            LatentSize = latentSize;
            DecoderHidden = decoderHidden;

            _enc1 = new LSTMLayer(inputSize, encoderHidden, returnSequences: true);
            _enc2 = new LSTMLayer(encoderHidden, latentSize, returnSequences: false);
            _repeat = new RepeatVector(seqLen);
            _dec1 = new LSTMLayer(latentSize, decoderHidden, returnSequences: true);
            _dec2 = new LSTMLayer(decoderHidden, inputSize, returnSequences: true);
        }

        // ---------------------------------------------------------------------------
        // Train / Eval
        // ---------------------------------------------------------------------------

        public void Train()
        {
            IsTraining = true;
            _enc1.Train();
            _enc2.Train();
            _dec1.Train();
            _dec2.Train();
        }

        public void Eval()
        {
            IsTraining = false;
            _enc1.Eval();
            _enc2.Eval();
            _dec1.Eval();
            _dec2.Eval();
        }

        // ---------------------------------------------------------------------------
        // Forward — training path with autograd
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Full forward pass with computation graph recording.
        /// input shape: [batch, seqLen, inputSize]
        /// output shape: [batch, seqLen, inputSize]
        /// </summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var latent = Encode(graph, input);
            return Decode(graph, latent);
        }

        // ---------------------------------------------------------------------------
        // Inference — reconstruction error per window
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Returns mean squared reconstruction error for each window in the batch.
        /// Call Eval() before using in production.
        /// input shape: [batch, seqLen, inputSize]
        /// output: float[batch] — one MSE per window
        /// </summary>
        public float[] ReconstructionError(AutogradNode input)
        {
            var recon = Forward(null, input);
            var batch = input.Data.GetDim(0);
            var n = SeqLen * InputSize;
            var errors = new float[batch];

            var srcS = input.Data.AsReadOnlySpan();
            var recS = recon.Data.AsReadOnlySpan();

            using var diffBuf = new FastBuffer<float>(n);
            var diffS = diffBuf.AsSpan();

            for (var b = 0; b < batch; b++)
            {
                var src = srcS.Slice(b * n, n);
                var rec = recS.Slice(b * n, n);

                System.Numerics.Tensors.TensorPrimitives.Subtract(src, rec, diffS);
                errors[b] = System.Numerics.Tensors.TensorPrimitives.Dot(
                    (ReadOnlySpan<float>)diffS,
                    (ReadOnlySpan<float>)diffS) / n;
            }

            return errors;
        }

        // ---------------------------------------------------------------------------
        // Encode / Decode — exposed for inspection
        // ---------------------------------------------------------------------------

        /// <summary>Encoder only — returns latent vector [batch, latentSize].</summary>
        public AutogradNode Encode(ComputationGraph graph, AutogradNode input)
        {
            var h1 = _enc1.Forward(graph, input);
            return _enc2.Forward(graph, h1);
        }

        /// <summary>Decoder only — takes latent [batch, latentSize], returns [batch, seqLen, inputSize].</summary>
        public AutogradNode Decode(ComputationGraph graph, AutogradNode latent)
        {
            var repeated = _repeat.Forward(graph, latent);
            var h1 = _dec1.Forward(graph, repeated);
            return _dec2.Forward(graph, h1);
        }

        // ---------------------------------------------------------------------------
        // IModule
        // ---------------------------------------------------------------------------

        public IEnumerable<AutogradNode> Parameters()
        {
            foreach (var p in _enc1.Parameters()) yield return p;
            foreach (var p in _enc2.Parameters()) yield return p;
            foreach (var p in _dec1.Parameters()) yield return p;
            foreach (var p in _dec2.Parameters()) yield return p;
        }

        public int ParameterCount => Parameters().Sum(p => p.Data.Size);

        public void Save(BinaryWriter bw)
        {
            _enc1.Save(bw);
            _enc2.Save(bw);
            _dec1.Save(bw);
            _dec2.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            _enc1.Load(br);
            _enc2.Load(br);
            _dec1.Load(br);
            _dec2.Load(br);
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
                throw new FileNotFoundException($"Model file not found: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);
            Load(br);
        }

        public void Dispose()
        {
            _enc1.Dispose();
            _enc2.Dispose();
            _repeat.Dispose();
            _dec1.Dispose();
            _dec2.Dispose();
        }
    }
}