// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    ///     LSTM Autoencoder for time-series anomaly detection.
    ///     Includes a strictly 0-allocation inference path for real-time K8s monitoring.
    /// </summary>
    public sealed class LSTMAutoencoder : IModule
    {
        private readonly LSTMLayer _dec1;
        private readonly LSTMLayer _dec2;
        private readonly LSTMLayer _enc1;
        private readonly LSTMLayer _enc2;
        private readonly RepeatVector _repeat;

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

            _enc1 = new LSTMLayer(inputSize, encoderHidden, true);
            _enc2 = new LSTMLayer(encoderHidden, latentSize, false);
            _repeat = new RepeatVector(seqLen);
            _dec1 = new LSTMLayer(latentSize, decoderHidden, true);
            _dec2 = new LSTMLayer(decoderHidden, inputSize, true);
        }

        public int InputSize { get; }
        public int SeqLen { get; }
        public int EncoderHidden { get; }
        public int LatentSize { get; }
        public int DecoderHidden { get; }

        public int ParameterCount => Parameters().Sum(p => p.DataView.Size);

        public bool IsTraining { get; private set; } = true;

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
        // Forward — Training Path (Autograd)
        // ---------------------------------------------------------------------------

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            // Adapter dla standardowego interfejsu IModule. 
            // Wymusza predykcję dla pojedynczej sekwencji (Batch = 1).
            Reconstruct(1, input, output);
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var latent = Encode(graph, input);
            return Decode(graph, latent);
        }

        // ---------------------------------------------------------------------------
        // IModule
        // ---------------------------------------------------------------------------

        public IEnumerable<AutogradNode> Parameters()
        {
            foreach (var p in _enc1.Parameters())
            {
                yield return p;
            }
            foreach (var p in _enc2.Parameters())
            {
                yield return p;
            }
            foreach (var p in _dec1.Parameters())
            {
                yield return p;
            }
            foreach (var p in _dec2.Parameters())
            {
                yield return p;
            }
        }

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

        public void Dispose()
        {
            _enc1.Dispose();
            _enc2.Dispose();
            _repeat.Dispose();
            _dec1.Dispose();
            _dec2.Dispose();
        }

        public AutogradNode Encode(ComputationGraph graph, AutogradNode input)
        {
            var h1 = _enc1.Forward(graph, input);
            return _enc2.Forward(graph, h1);
        }

        public AutogradNode Decode(ComputationGraph graph, AutogradNode latent)
        {
            var repeated = _repeat.Forward(graph, latent);
            var h1 = _dec1.Forward(graph, repeated);
            return _dec2.Forward(graph, h1);
        }

        // ---------------------------------------------------------------------------
        // Inference — Zero-Allocation Fast Path
        // ---------------------------------------------------------------------------

        /// <summary>
        ///     100% Allocation-free reconstruction path.
        /// </summary>
        public void Reconstruct(int batchSize, ReadOnlySpan<float> features, Span<float> reconstruction)
        {
            var n = SeqLen * InputSize;

            if (features.Length != batchSize * n)
            {
                throw new ArgumentException($"Expected feature length {batchSize * n}, got {features.Length}.");
            }

            var enc1Len = batchSize * SeqLen * EncoderHidden;
            var latentLen = batchSize * LatentSize;
            var repLen = batchSize * SeqLen * LatentSize;
            var dec1Len = batchSize * SeqLen * DecoderHidden;

            // Brak wielkich bloków try-finally, wszystko załatwiają PooledBuffery
            using var enc1Buf = new PooledBuffer<float>(enc1Len);
            using var latentBuf = new PooledBuffer<float>(latentLen);
            using var repBuf = new PooledBuffer<float>(repLen);
            using var dec1Buf = new PooledBuffer<float>(dec1Len);

            var enc1S = enc1Buf.Span;
            var latentS = latentBuf.Span;
            var repS = repBuf.Span;
            var dec1S = dec1Buf.Span;

            // 1. Encoder 1
            _enc1.ForwardInference(batchSize, SeqLen, features, enc1S);

            // 2. Encoder 2
            _enc2.ForwardInference(batchSize, SeqLen, enc1S, latentS);

            // 3. Repeat Vector (Manual copy to avoid allocations)
            for (var b = 0; b < batchSize; b++)
            {
                var src = latentS.Slice(b * LatentSize, LatentSize);

                for (var t = 0; t < SeqLen; t++)
                {
                    src.CopyTo(repS.Slice(b * SeqLen * LatentSize + t * LatentSize, LatentSize));
                }
            }

            // 4. Decoder 1
            _dec1.ForwardInference(batchSize, SeqLen, repS, dec1S);

            // 5. Decoder 2
            _dec2.ForwardInference(batchSize, SeqLen, dec1S, reconstruction);
        }

        /// <summary>
        ///     Returns MSE reconstruction error per-window for anomaly thresholding.
        ///     Triggers zero GC allocations (except the returned float[] result array).
        /// </summary>
        public float[] ReconstructionError(ReadOnlySpan<float> input, int batchSize = 1)
        {
            var n = SeqLen * InputSize;
            var errors = new float[batchSize];

            using var reconBuf = new PooledBuffer<float>(batchSize * n);
            using var diffBuf = new PooledBuffer<float>(n);

            var reconS = reconBuf.Span;
            var diffS = diffBuf.Span;

            Reconstruct(batchSize, input, reconS);

            for (var b = 0; b < batchSize; b++)
            {
                var src = input.Slice(b * n, n);
                var rec = reconS.Slice(b * n, n);

                TensorPrimitives.Subtract(src, rec, diffS);

                errors[b] = TensorPrimitives.Dot(diffS, diffS) / n;
            }

            return errors;
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
    }
}