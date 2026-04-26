// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Evolutionary.Adapters;

namespace DevOnBike.Overfit.Demo.Unity.Server
{
    /// <summary>
    ///     Neural policy driving a single bot: a two-layer MLP with tanh activations
    ///     (<c>inputSize → hiddenSize → outputSize</c>). Each layer is an Overfit
    ///     <see cref="IModule"/>; the <see cref="NeuralNetworkParameterAdapter"/> bridges
    ///     a single flat evolutionary genome to all the per-layer parameter tensors.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The forward pipeline is composed manually rather than through
    ///         <see cref="Sequential"/>. Reason: <see cref="Sequential.ForwardInference"/>
    ///         forwards its internal 64 K scratch buffer directly to nested layers, so a
    ///         <see cref="LinearLayer"/> that treats <c>output.Length</c> as its output
    ///         dimension ends up trying to write 64 K outputs per call. A hand-rolled
    ///         pipeline here sizes the per-layer buffers exactly (4 → 16 → 2) and keeps
    ///         the hot path allocation-free after warmup.
    ///     </para>
    ///     <para>
    ///         Not thread-safe (the hidden buffer is reused across calls). For parallel
    ///         fitness evaluation, build one <see cref="SwarmBrain"/> per worker thread.
    ///     </para>
    /// </remarks>
    public sealed class SwarmBrain : IDisposable
    {
        // The module stack is exposed to the adapter via a tiny Sequential wrapper — the
        // adapter walks IModule.Parameters() recursively, so Sequential is a convenient
        // container even though we don't use its ForwardInference.
        private readonly Sequential _parameterContainer;
        private readonly NeuralNetworkParameterAdapter _adapter;

        private readonly LinearLayer _hiddenLinear;
        private readonly TanhActivation _hiddenActivation;
        private readonly LinearLayer _outputLinear;
        private readonly TanhActivation _outputActivation;

        // Scratch buffers sized to exactly what each layer needs.
        private readonly float[] _hiddenPreActivation;
        private readonly float[] _hiddenPostActivation;
        private readonly float[] _outputPreActivation;

        public SwarmBrain(int inputSize, int hiddenSize, int outputSize)
        {
            if (inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(inputSize), "Layer sizes must be positive.");
            }

            _hiddenLinear = new LinearLayer(inputSize, hiddenSize);
            _hiddenActivation = new TanhActivation();
            _outputLinear = new LinearLayer(hiddenSize, outputSize);
            _outputActivation = new TanhActivation();

            _parameterContainer = new Sequential(
                _hiddenLinear,
                _hiddenActivation,
                _outputLinear,
                _outputActivation);

            _parameterContainer.Eval();

            _adapter = new NeuralNetworkParameterAdapter(_parameterContainer);

            _hiddenPreActivation = new float[hiddenSize];
            _hiddenPostActivation = new float[hiddenSize];
            _outputPreActivation = new float[outputSize];

            InputSize = inputSize;
            HiddenSize = hiddenSize;
            OutputSize = outputSize;
        }

        public int InputSize { get; }

        public int HiddenSize { get; }

        public int OutputSize { get; }

        public int ParameterCount => _adapter.ParameterCount;

        /// <summary>
        ///     Runs a forward pass over a single observation. Zero allocation after warmup.
        /// </summary>
        public void Infer(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != InputSize)
            {
                throw new ArgumentException($"input must have length {InputSize}, got {input.Length}.");
            }

            if (output.Length != OutputSize)
            {
                throw new ArgumentException($"output must have length {OutputSize}, got {output.Length}.");
            }

            // 4 → 16 (linear) → 16 (tanh) → 2 (linear) → 2 (tanh, into user's output buffer).
            _hiddenLinear.ForwardInference(input, _hiddenPreActivation);
            _hiddenActivation.ForwardInference(_hiddenPreActivation, _hiddenPostActivation);
            _outputLinear.ForwardInference(_hiddenPostActivation, _outputPreActivation);
            _outputActivation.ForwardInference(_outputPreActivation, output);
        }

        /// <summary>
        ///     Copies a candidate genome into the network's parameters. The genome must have
        ///     exactly <see cref="ParameterCount"/> entries in the adapter's parameter order.
        /// </summary>
        public void LoadGenome(ReadOnlySpan<float> genome)
        {
            _adapter.ReadFromVector(genome);
        }

        /// <summary>
        ///     Reads the network's current parameters into <paramref name="genome"/>.
        /// </summary>
        public void StoreGenome(Span<float> genome)
        {
            _adapter.WriteToVector(genome);
        }

        /// <summary>
        ///     Writes the network's parameters as a flat little-endian <c>float32</c> blob.
        ///     Matches the Unity client's expected format.
        /// </summary>
        public void SaveToFile(string path)
        {
            var genome = new float[ParameterCount];
            StoreGenome(genome);

            var bytes = new byte[ParameterCount * sizeof(float)];
            MemoryMarshal.AsBytes(genome.AsSpan()).CopyTo(bytes);
            File.WriteAllBytes(path, bytes);
        }

        /// <summary>
        ///     Reads a <c>ParameterCount × float32</c> blob and loads it into the network.
        ///     Returns false and leaves the network untouched when the file is missing, has
        ///     the wrong size, or contains non-finite values.
        /// </summary>
        public bool LoadFromFile(string path)
        {
            if (!File.Exists(path))
            {
                return false;
            }

            var bytes = File.ReadAllBytes(path);
            var expectedBytes = ParameterCount * sizeof(float);

            if (bytes.Length != expectedBytes)
            {
                Console.Error.WriteLine(
                    $"[SwarmBrain] Brain file size mismatch: {bytes.Length} vs expected {expectedBytes}. " +
                    "This can happen after changing the network architecture; retrain from scratch.");
                return false;
            }

            var genome = new float[ParameterCount];
            Buffer.BlockCopy(bytes, 0, genome, 0, bytes.Length);

            foreach (var w in genome)
            {
                if (!float.IsFinite(w))
                {
                    Console.Error.WriteLine("[SwarmBrain] Brain file contains NaN/Infinity. Rejecting.");
                    return false;
                }
            }

            LoadGenome(genome);
            return true;
        }

        /// <summary>
        ///     Static helper so <see cref="SwarmConfig"/> can compute <c>GenomeSize</c> without
        ///     instantiating a network. Kept in sync with the architecture built by the
        ///     constructor: two LinearLayers with their biases, activations have no parameters.
        /// </summary>
        public static int CountParameters(int inputSize, int hiddenSize, int outputSize)
        {
            // LinearLayer params = in*out (weights) + out (biases).
            var hidden = (inputSize * hiddenSize) + hiddenSize;
            var output = (hiddenSize * outputSize) + outputSize;
            return hidden + output;
        }

        public void Dispose()
        {
            _parameterContainer.Dispose();
        }
    }
}