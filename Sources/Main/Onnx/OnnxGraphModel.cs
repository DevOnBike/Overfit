// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;

namespace DevOnBike.Overfit.Onnx
{
    /// <summary>
    /// Executes an ONNX model with arbitrary DAG topology (including skip connections).
    ///
    /// Execution plan:
    ///   - Nodes are topologically sorted at import time.
    ///   - Each node has a fixed input slot (or two for Add) and output slot.
    ///   - Intermediate buffers are pre-allocated once at construction.
    ///   - Inference: iterates nodes in order, calling ForwardInference per node.
    ///
    /// Buffer layout:
    ///   Slot 0      = model input (written by caller).
    ///   Slot 1..N   = intermediate activations (written by nodes, freed when no
    ///                 longer needed — future optimisation).
    ///   Last slot   = model output (read by caller after forward).
    /// </summary>
    public sealed class OnnxGraphModel : IDisposable
    {
        private readonly OnnxGraphNode[] _nodes;
        private readonly float[][] _buffers;
        private readonly int _inputSize;
        private readonly int _outputSize;
        private bool _disposed;

        internal OnnxGraphModel(
            OnnxGraphNode[] nodes,
            float[][] buffers,
            int inputSize,
            int outputSize)
        {
            _nodes      = nodes;
            _buffers    = buffers;
            _inputSize  = inputSize;
            _outputSize = outputSize;
        }

        public int InputSize  => _inputSize;
        public int OutputSize => _outputSize;

        /// <summary>
        /// Runs a single-sample inference through the DAG.
        /// Input must be <see cref="InputSize"/> floats.
        /// Output receives <see cref="OutputSize"/> floats.
        /// </summary>
        public void RunInference(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != _inputSize)
            {
                throw new ArgumentException(
                    $"Expected input size {_inputSize}, got {input.Length}.",
                    nameof(input));
            }

            if (output.Length < _outputSize)
            {
                throw new ArgumentException(
                    $"Output span must be at least {_outputSize}, got {output.Length}.",
                    nameof(output));
            }

            // Slot 0 = model input.
            input.CopyTo(_buffers[0]);

            foreach (var node in _nodes)
            {
                var outBuf = _buffers[node.OutputSlot].AsSpan(0, node.OutputSize);

                if (node.InputSlots.Length == 2 && node.Module is OnnxAddLayer addLayer)
                {
                    // Skip connection: Add(left, right) → output.
                    var left  = _buffers[node.InputSlots[0]].AsSpan();
                    var right = _buffers[node.InputSlots[1]].AsSpan();
                    addLayer.ForwardInference(left, right, outBuf);
                }
                else
                {
                    var inBuf = _buffers[node.InputSlots[0]].AsSpan();
                    node.Module.ForwardInference(inBuf, outBuf);
                }
            }

            // Last node's output slot → caller's output span.
            var lastNode = _nodes[^1];
            _buffers[lastNode.OutputSlot].AsSpan(0, _outputSize).CopyTo(output);
        }

        /// <summary>
        /// Sets all modules to evaluation mode (uses running stats for BatchNorm, etc.).
        /// Should be called before inference.
        /// </summary>
        public void Eval()
        {
            foreach (var node in _nodes)
            {
                node.Module.Eval();
            }
        }

        /// <summary>
        /// Sets all modules to training mode.
        /// </summary>
        public void Train()
        {
            foreach (var node in _nodes)
            {
                node.Module.Train();
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;

            foreach (var node in _nodes)
            {
                node.Module.Dispose();
            }
        }
    }
}
