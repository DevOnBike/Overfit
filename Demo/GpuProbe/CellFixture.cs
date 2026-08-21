// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Everything one cell needs on the host at ONE token count: synthetic activations, the output and
    /// gradient buffers the CPU arms write, and the two computation graphs the real training op runs on.
    /// The weight itself lives in <see cref="CellWeights"/> and is shared across the batch sweep.
    /// </summary>
    internal sealed class CellFixture : IDisposable
    {
        private readonly ComputationGraph _forwardGraph;
        private readonly ComputationGraph _backwardGraph;
        private readonly TensorStorage<float> _inputStorage;
        private readonly TensorStorage<float> _backwardInputStorage;
        private readonly AutogradNode _inputNode;
        private readonly AutogradNode _backwardInputNode;
        private readonly AutogradNode _backwardOutputNode;
        private readonly float[] _partials;
        private readonly int _partitions;

        public CellFixture(CellWeights weights, int n, int seed)
        {
            Weights = weights;
            Cell = weights.Cell;
            N = n;

            var cell = weights.Cell;
            var k = cell.K;
            var m = cell.M;

            // Offset from the weight's own seed so the activations are not correlated with it.
            var rnd = new Random(seed + 1);

            Input = new float[(long)n * k];
            for (var i = 0; i < Input.Length; i++)
            {
                Input[i] = (float)(rnd.NextDouble() * 2 - 1);
            }

            OutputGrad = new float[(long)n * m];
            for (var i = 0; i < OutputGrad.Length; i++)
            {
                OutputGrad[i] = (float)(rnd.NextDouble() * 2 - 1);
            }

            OutputF32 = new float[(long)n * m];
            InputGradF32 = new float[(long)n * k];

            // Arm C1 keeps one output temporary live at a time and returns the arena after every
            // repetition, so one output plus slack is enough. Arm C2 keeps its output for the whole cell
            // and needs a gradient beside it, hence twice.
            _forwardGraph = new ComputationGraph(ArenaElements((long)n * m + 1024));
            _backwardGraph = new ComputationGraph(ArenaElements((2L * n * m) + 1024));

            _inputStorage = new TensorStorage<float>(n * k, clearMemory: false);
            Input.CopyTo(_inputStorage.AsSpan());
            _inputNode = new AutogradNode(_inputStorage, new TensorShape(n, k), requiresGrad: false);

            _backwardInputStorage = new TensorStorage<float>(n * k, clearMemory: false);
            Input.CopyTo(_backwardInputStorage.AsSpan());
            _backwardInputNode = new AutogradNode(_backwardInputStorage, new TensorShape(n, k), requiresGrad: true);

            // One recorded forward, outside every timed region: arm C2 then replays only the backward.
            _backwardOutputNode = _backwardGraph.FrozenQuantizedLinear(_backwardInputNode, Quantized);

            _partitions = Math.Min(OverfitParallel.WorkerCount, m);

            var partialElements = (long)_partitions * n * k;
            if (partialElements > int.MaxValue)
            {
                throw new InvalidOperationException(
                    $"cell {cell.Name} at n={n} needs {partialElements} partial-gradient elements (> int.MaxValue).");
            }

            _partials = new float[partialElements];
        }

        public Cell Cell { get; }

        public int N { get; }

        public CellWeights Weights { get; }

        /// <summary>The dequantized weight — what the device gets and what arms C3 and C4 read.</summary>
        public float[] WeightF32 => Weights.F32;

        public Q4KWeight Quantized => Weights.Quantized;

        public float[] Input { get; }

        public float[] OutputGrad { get; }

        /// <summary>Destination of arm C3. Also the parity reference for the device forward arms.</summary>
        public float[] OutputF32 { get; }

        /// <summary>Destination of arm C4. Also the parity reference for the device backward arm.</summary>
        public float[] InputGradF32 { get; }

        /// <summary>C1 — the real training op: dequantize each Q4_K row once, then F32 dots across the batch.</summary>
        public void RunC1Forward() => _forwardGraph.FrozenQuantizedLinear(_inputNode, Quantized);

        /// <summary>Returns C1's arena. Outside the clock.</summary>
        public void ResetC1() => _forwardGraph.Reset();

        /// <summary>C2 — the real backward, dInput only. Includes the dequantize that G3 does not perform.</summary>
        public void RunC2Backward() => _backwardGraph.BackwardFromGrad(_backwardOutputNode);

        /// <summary>Seeds the output gradient and clears the input gradient. Outside the clock.</summary>
        public void SeedC2()
        {
            _backwardInputNode.ZeroGrad();
            OutputGrad.CopyTo(_backwardOutputNode.GradView.AsSpan());
        }

        /// <summary>Runs C1 once and copies its result out. Used by the parity check, never timed.</summary>
        public void ReadC1Output(Span<float> destination)
        {
            var node = _forwardGraph.FrozenQuantizedLinear(_inputNode, Quantized);
            node.DataView.AsReadOnlySpan().CopyTo(destination);
            _forwardGraph.Reset();
        }

        public void ReadC2InputGrad(Span<float> destination) =>
            _backwardInputNode.GradView.AsReadOnlySpan().CopyTo(destination);

        /// <summary>
        /// C3 — the same weight-stationary GEMM with the dequantize removed. Partitioned over output rows
        /// exactly the way <c>FrozenQuantizedLinear</c>'s own parallel forward is, so C1 minus C3 isolates
        /// the dequantize and not a difference in how the work was split.
        /// </summary>
        public void RunC3Forward()
        {
            int n = N, k = Cell.K, m = Cell.M;
            var p = Math.Min(OverfitParallel.WorkerCount, m);
            var weight = WeightF32;
            var input = Input;
            var output = OutputF32;

            OverfitParallel.For(0, p, partition =>
            {
                var start = (int)((long)partition * m / p);
                var end = (int)((long)(partition + 1) * m / p);
                for (var o = start; o < end; o++)
                {
                    var row = new ReadOnlySpan<float>(weight, o * k, k);
                    for (var b = 0; b < n; b++)
                    {
                        output[b * m + o] = TensorPrimitives.Dot(new ReadOnlySpan<float>(input, b * k, k), row);
                    }
                }
            });
        }

        /// <summary>
        /// C4 — the dequantize-free backward. NOT one of the seven arms in the signed plan; added so the
        /// device backward G3 has a counterpart that performs the same work it does. Arm C2 includes a
        /// dequantize that G3 never runs, so C2 against G3 would repeat, in the backward direction, the
        /// unfair comparison the plan forbids in the forward one.
        /// Partitioned like the real backward: private partial gradients, reduced at the end.
        /// </summary>
        public void RunC4Backward()
        {
            int n = N, k = Cell.K, m = Cell.M;
            var p = _partitions;
            var weight = WeightF32;
            var dy = OutputGrad;
            var partials = _partials;
            var dx = InputGradF32;

            Array.Clear(partials);
            Array.Clear(dx);

            OverfitParallel.For(0, p, partition =>
            {
                var oStart = (int)((long)partition * m / p);
                var oEnd = (int)((long)(partition + 1) * m / p);
                var slot = partition * n * k;
                for (var o = oStart; o < oEnd; o++)
                {
                    var row = new ReadOnlySpan<float>(weight, o * k, k);
                    for (var b = 0; b < n; b++)
                    {
                        var g = dy[b * m + o];
                        if (g == 0f)
                        {
                            continue;
                        }

                        var target = new Span<float>(partials, slot + b * k, k);
                        TensorPrimitives.MultiplyAdd(row, g, target, target);
                    }
                }
            });

            for (var partition = 0; partition < p; partition++)
            {
                TensorPrimitives.Add(
                    dx.AsSpan(),
                    new ReadOnlySpan<float>(partials, partition * n * k, n * k),
                    dx.AsSpan());
            }
        }

        public void Dispose()
        {
            // _backwardOutputNode is a GraphTemporary: the graph owns it and disposes it. Disposing it
            // here would be a second dispose of storage the arena already reclaimed.
            _backwardInputNode.Dispose();
            _inputNode.Dispose();
            _backwardInputStorage.Dispose();
            _inputStorage.Dispose();
            _backwardGraph.Dispose();
            _forwardGraph.Dispose();
        }

        private static int ArenaElements(long wanted)
        {
            if (wanted > int.MaxValue)
            {
                throw new InvalidOperationException($"graph arena would need {wanted} elements (> int.MaxValue).");
            }

            return (int)wanted;
        }
    }
}
