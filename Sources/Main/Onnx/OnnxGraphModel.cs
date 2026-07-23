// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Text;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors.Core;

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
        private readonly TensorStorage<float>[] _buffers;
        private readonly int _inputSize;
        private readonly int _outputSize;
        private bool _disposed;

        internal OnnxGraphModel(
            OnnxGraphNode[] nodes,
            TensorStorage<float>[] buffers,
            int inputSize,
            int outputSize)
        {
            _nodes = nodes;
            _buffers = buffers;
            _inputSize = inputSize;
            _outputSize = outputSize;
        }

        public int InputSize => _inputSize;
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
            input.CopyTo(_buffers[0].AsSpan());

            for (var i = 0; i < _nodes.Length; i++)
            {
                var node = _nodes[i];
                var started = ProfileNodes ? Stopwatch.GetTimestamp() : 0L;
                var outBuf = _buffers[node.OutputSlot].AsSpan().Slice(0, node.OutputSize);

                if (node.InputSlots.Length == 2 && node.Module is OnnxAddLayer addLayer)
                {
                    // Skip connection: Add(left, right) → output.
                    var left = _buffers[node.InputSlots[0]].AsSpan();
                    var right = _buffers[node.InputSlots[1]].AsSpan();
                    addLayer.ForwardInference(left, right, outBuf);
                }

                if (!(node.InputSlots.Length == 2 && node.Module is OnnxAddLayer))
                {
                    var inBuf = _buffers[node.InputSlots[0]].AsSpan();
                    node.Module.ForwardInference(inBuf, outBuf);
                }

                if (ProfileNodes)
                {
                    RecordNode(i, Stopwatch.GetTimestamp() - started);
                }
            }

            // Last node's output slot → caller's output span.
            var lastNode = _nodes[^1];
            _buffers[lastNode.OutputSlot].AsSpan().Slice(0, _outputSize).CopyTo(output);
        }

        /// <summary>
        /// Opt-in per-node timing. Off by default and checked before any timestamp is taken, so the inference
        /// path is unchanged when it is off.
        ///
        /// <para>This is the CNN counterpart of <c>PrefillProfiler</c>, and exists for the same reason: after
        /// parallelising the im2col gather, VGG-16's Amdahl serial fraction fell from ~15.5% to ~7.3% — but at
        /// 32 workers that residue still <i>dominates</i> (~49 of 73 ms). Guessing which operator holds it
        /// would be guessing about mechanism, which this project has been wrong about repeatedly; this
        /// measures it per operator instead.</para>
        /// </summary>
        public static bool ProfileNodes;

        private long[]? _nodeTicks;
        private long[]? _nodeCalls;

        private void RecordNode(int index, long ticks)
        {
            _nodeTicks ??= new long[_nodes.Length];
            _nodeCalls ??= new long[_nodes.Length];

            _nodeTicks[index] += ticks;
            _nodeCalls[index]++;
        }

        /// <summary>
        /// Every node individually — index, operator, output size and ms — rather than grouped by operator.
        ///
        /// <para>Grouping answers "which operator"; this answers "which <i>layer</i>", which is the question
        /// once an operator's cost is known to be shape-dependent. On VGG-16 a standalone prototype of the conv
        /// GEMM reached 132 GFLOP/s single-threaded on a late-layer shape while production conv averaged
        /// 19.7 GFLOP/s across all layers — a 6.7× spread that only a per-layer view can locate.</para>
        /// </summary>
        public string PerNodeProfileReport()
        {
            if (_nodeTicks is null)
            {
                return "(no node profile recorded — set OnnxGraphModel.ProfileNodes before running)";
            }

            var toMs = 1000.0 / Stopwatch.Frequency;
            var runs = _nodeCalls is null || _nodeCalls.Length == 0 ? 1L : Math.Max(1L, _nodeCalls[0]);
            var sb = new StringBuilder();

            sb.AppendLine($"=== per-node ({runs} run(s)) ===");

            for (var i = 0; i < _nodes.Length; i++)
            {
                var node = _nodes[i];
                sb.AppendLine(
                    $"  [{i,2}] {node.Module.GetType().Name,-26} out={node.OutputSize,9}  {_nodeTicks[i] * toMs / runs,8:F2} ms");
            }

            return sb.ToString();
        }

        /// <summary>Clears the per-node accumulators (call before the measured segment).</summary>
        public void ResetNodeProfile()
        {
            _nodeTicks = null;
            _nodeCalls = null;
        }

        /// <summary>
        /// Per-operator totals, heaviest first: total ms, share of the measured wall time, and node count.
        /// Grouped by module type, because "which operator" is the actionable unit — not which of 40 nodes.
        /// </summary>
        public string NodeProfileReport()
        {
            if (_nodeTicks is null)
            {
                return "(no node profile recorded — set OnnxGraphModel.ProfileNodes before running)";
            }

            // Aggregate into parallel arrays and selection-sort them. A Dictionary + OrderByDescending would
            // read better, but LINQ is banned in Sources/Main (RS0030) and the node count is tiny, so an
            // O(n^2) sort over distinct operator names costs nothing.
            var names = new string[_nodes.Length];
            var ticks = new long[_nodes.Length];
            var counts = new int[_nodes.Length];
            var distinct = 0;
            var total = 0L;

            for (var i = 0; i < _nodes.Length; i++)
            {
                var name = _nodes[i].Module.GetType().Name;
                total += _nodeTicks[i];

                var slot = -1;
                for (var j = 0; j < distinct; j++)
                {
                    if (string.Equals(names[j], name, StringComparison.Ordinal))
                    {
                        slot = j;
                        break;
                    }
                }

                if (slot < 0)
                {
                    slot = distinct;
                    names[slot] = name;
                    distinct++;
                }

                ticks[slot] += _nodeTicks[i];
                counts[slot]++;
            }

            for (var a = 0; a < distinct - 1; a++)
            {
                var best = a;
                for (var bIdx = a + 1; bIdx < distinct; bIdx++)
                {
                    if (ticks[bIdx] > ticks[best])
                    {
                        best = bIdx;
                    }
                }

                (names[a], names[best]) = (names[best], names[a]);
                (ticks[a], ticks[best]) = (ticks[best], ticks[a]);
                (counts[a], counts[best]) = (counts[best], counts[a]);
            }

            var toMs = 1000.0 / Stopwatch.Frequency;
            var runs = _nodeCalls is null || _nodeCalls.Length == 0 ? 1L : Math.Max(1L, _nodeCalls[0]);
            var sb = new StringBuilder();

            sb.AppendLine($"=== OnnxGraphModel node profile ({runs} run(s), {_nodes.Length} nodes) ===");
            sb.AppendLine($"  total: {total * toMs / runs:F2} ms/run");

            for (var i = 0; i < distinct; i++)
            {
                var ms = ticks[i] * toMs / runs;
                sb.AppendLine(
                    $"  {names[i],-28} {ms,8:F2} ms {(total == 0 ? 0 : 100.0 * ticks[i] / total),6:F1}%   ({counts[i]} nodes)");
            }

            return sb.ToString();
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

            foreach (var buf in _buffers)
            {
                buf.Dispose();
            }
        }
    }
}
