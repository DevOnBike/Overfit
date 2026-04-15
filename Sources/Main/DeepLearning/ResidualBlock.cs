using System.Diagnostics;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class ResidualBlock : IModule
    {
        private readonly BatchNorm1D _bn1;
        private readonly BatchNorm1D _bn2;
        private readonly LinearLayer _linear1;
        private readonly LinearLayer _linear2;

        public ResidualBlock(int hiddenSize)
        {
            _linear1 = new LinearLayer(hiddenSize, hiddenSize);
            _bn1 = new BatchNorm1D(hiddenSize);
            _linear2 = new LinearLayer(hiddenSize, hiddenSize);
            _bn2 = new BatchNorm1D(hiddenSize);
        }

        public bool IsTraining { get; private set; } = true;

        public static bool DiagnosticsEnabled { get; set; }

        private static long _calls;
        private static long _totalTicks;
        private static long _totalAllocBytes;

        private static long _linear1Ticks;
        private static long _linear1AllocBytes;

        private static long _bn1Ticks;
        private static long _bn1AllocBytes;

        private static long _relu1Ticks;
        private static long _relu1AllocBytes;

        private static long _linear2Ticks;
        private static long _linear2AllocBytes;

        private static long _bn2Ticks;
        private static long _bn2AllocBytes;

        private static long _addTicks;
        private static long _addAllocBytes;

        private static long _relu2Ticks;
        private static long _relu2AllocBytes;

        public static void ResetDiagnostics()
        {
            Interlocked.Exchange(ref _calls, 0);
            Interlocked.Exchange(ref _totalTicks, 0);
            Interlocked.Exchange(ref _totalAllocBytes, 0);

            Interlocked.Exchange(ref _linear1Ticks, 0);
            Interlocked.Exchange(ref _linear1AllocBytes, 0);

            Interlocked.Exchange(ref _bn1Ticks, 0);
            Interlocked.Exchange(ref _bn1AllocBytes, 0);

            Interlocked.Exchange(ref _relu1Ticks, 0);
            Interlocked.Exchange(ref _relu1AllocBytes, 0);

            Interlocked.Exchange(ref _linear2Ticks, 0);
            Interlocked.Exchange(ref _linear2AllocBytes, 0);

            Interlocked.Exchange(ref _bn2Ticks, 0);
            Interlocked.Exchange(ref _bn2AllocBytes, 0);

            Interlocked.Exchange(ref _addTicks, 0);
            Interlocked.Exchange(ref _addAllocBytes, 0);

            Interlocked.Exchange(ref _relu2Ticks, 0);
            Interlocked.Exchange(ref _relu2AllocBytes, 0);
        }

        public static ResidualBlockDiagnosticsSnapshot GetDiagnosticsSnapshot()
        {
            return new ResidualBlockDiagnosticsSnapshot(
                Calls: Interlocked.Read(ref _calls),
                TotalMs: TicksToMs(Interlocked.Read(ref _totalTicks)),
                TotalAllocBytes: Interlocked.Read(ref _totalAllocBytes),
                Linear1Ms: TicksToMs(Interlocked.Read(ref _linear1Ticks)),
                Linear1AllocBytes: Interlocked.Read(ref _linear1AllocBytes),
                BatchNorm1Ms: TicksToMs(Interlocked.Read(ref _bn1Ticks)),
                BatchNorm1AllocBytes: Interlocked.Read(ref _bn1AllocBytes),
                ReLU1Ms: TicksToMs(Interlocked.Read(ref _relu1Ticks)),
                ReLU1AllocBytes: Interlocked.Read(ref _relu1AllocBytes),
                Linear2Ms: TicksToMs(Interlocked.Read(ref _linear2Ticks)),
                Linear2AllocBytes: Interlocked.Read(ref _linear2AllocBytes),
                BatchNorm2Ms: TicksToMs(Interlocked.Read(ref _bn2Ticks)),
                BatchNorm2AllocBytes: Interlocked.Read(ref _bn2AllocBytes),
                AddMs: TicksToMs(Interlocked.Read(ref _addTicks)),
                AddAllocBytes: Interlocked.Read(ref _addAllocBytes),
                ReLU2Ms: TicksToMs(Interlocked.Read(ref _relu2Ticks)),
                ReLU2AllocBytes: Interlocked.Read(ref _relu2AllocBytes));
        }

        public readonly record struct ResidualBlockDiagnosticsSnapshot(
            long Calls,
            double TotalMs,
            long TotalAllocBytes,
            double Linear1Ms,
            long Linear1AllocBytes,
            double BatchNorm1Ms,
            long BatchNorm1AllocBytes,
            double ReLU1Ms,
            long ReLU1AllocBytes,
            double Linear2Ms,
            long Linear2AllocBytes,
            double BatchNorm2Ms,
            long BatchNorm2AllocBytes,
            double AddMs,
            long AddAllocBytes,
            double ReLU2Ms,
            long ReLU2AllocBytes);

        private static double TicksToMs(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

        public void Train()
        {
            IsTraining = true;
            _linear1.Train();
            _bn1.Train();
            _linear2.Train();
            _bn2.Train();
        }

        public void Eval()
        {
            IsTraining = false;
            _linear1.Eval();
            _bn1.Eval();
            _linear2.Eval();
            _bn2.Eval();
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            var hiddenSize = _linear1.Weights.DataView.GetDim(0);

            using var buf1 = new PooledBuffer<float>(hiddenSize);
            using var buf2 = new PooledBuffer<float>(hiddenSize);

            var b1 = buf1.Span;
            var b2 = buf2.Span;

            _linear1.ForwardInference(input, b1);
            _bn1.ForwardInference(b1, b2);
            TensorPrimitives.Max(b2, 0f, b1);
            _linear2.ForwardInference(b1, b2);
            _bn2.ForwardInference(b2, b1);

            TensorPrimitives.Add(b1, input, output);
            TensorPrimitives.Max(output, 0f, output);
        }

        public AutogradNode Forward(ComputationGraph? graph, AutogradNode input)
        {
            if (!DiagnosticsEnabled)
            {
                return ForwardCore(graph, input);
            }

            Interlocked.Increment(ref _calls);

            long totalAllocBefore = GC.GetTotalAllocatedBytes(false);
            long totalStart = Stopwatch.GetTimestamp();

            try
            {
                return ForwardCoreWithDiagnostics(graph, input);
            }
            finally
            {
                long totalEnd = Stopwatch.GetTimestamp();
                long totalAllocAfter = GC.GetTotalAllocatedBytes(false);

                Interlocked.Add(ref _totalTicks, totalEnd - totalStart);
                Interlocked.Add(ref _totalAllocBytes, totalAllocAfter - totalAllocBefore);
            }
        }

        private AutogradNode ForwardCore(ComputationGraph? graph, AutogradNode input)
        {
            if (graph == null || !IsTraining)
            {
                var out1 = _linear1.Forward(null, input);
                using var bn1Out = _bn1.Forward(null, out1);
                using var a1 = TensorMath.ReLU(null, bn1Out);

                var out2 = _linear2.Forward(null, a1);
                using var bn2Out = _bn2.Forward(null, out2);

                using var added = TensorMath.Add(null, bn2Out, input);

                return TensorMath.ReLU(null, added);
            }

            var tOut1 = _linear1.Forward(graph, input);
            var tBn1 = _bn1.Forward(graph, tOut1);
            var tA1 = TensorMath.ReLU(graph, tBn1);

            var tOut2 = _linear2.Forward(graph, tA1);
            var tBn2 = _bn2.Forward(graph, tOut2);

            var tAdded = TensorMath.Add(graph, tBn2, input);

            return TensorMath.ReLU(graph, tAdded);
        }

        private AutogradNode ForwardCoreWithDiagnostics(ComputationGraph? graph, AutogradNode input)
        {
            long start;
            long allocBefore;
            long allocAfter;

            if (graph == null || !IsTraining)
            {
                allocBefore = GC.GetTotalAllocatedBytes(false);
                start = Stopwatch.GetTimestamp();
                var out1 = _linear1.Forward(null, input);
                Interlocked.Add(ref _linear1Ticks, Stopwatch.GetTimestamp() - start);
                allocAfter = GC.GetTotalAllocatedBytes(false);
                Interlocked.Add(ref _linear1AllocBytes, allocAfter - allocBefore);

                allocBefore = GC.GetTotalAllocatedBytes(false);
                start = Stopwatch.GetTimestamp();
                var bn1Out = _bn1.Forward(null, out1);
                Interlocked.Add(ref _bn1Ticks, Stopwatch.GetTimestamp() - start);
                allocAfter = GC.GetTotalAllocatedBytes(false);
                Interlocked.Add(ref _bn1AllocBytes, allocAfter - allocBefore);

                allocBefore = GC.GetTotalAllocatedBytes(false);
                start = Stopwatch.GetTimestamp();
                var a1 = TensorMath.ReLU(null, bn1Out);
                Interlocked.Add(ref _relu1Ticks, Stopwatch.GetTimestamp() - start);
                allocAfter = GC.GetTotalAllocatedBytes(false);
                Interlocked.Add(ref _relu1AllocBytes, allocAfter - allocBefore);

                allocBefore = GC.GetTotalAllocatedBytes(false);
                start = Stopwatch.GetTimestamp();
                var out2 = _linear2.Forward(null, a1);
                Interlocked.Add(ref _linear2Ticks, Stopwatch.GetTimestamp() - start);
                allocAfter = GC.GetTotalAllocatedBytes(false);
                Interlocked.Add(ref _linear2AllocBytes, allocAfter - allocBefore);

                allocBefore = GC.GetTotalAllocatedBytes(false);
                start = Stopwatch.GetTimestamp();
                var bn2Out = _bn2.Forward(null, out2);
                Interlocked.Add(ref _bn2Ticks, Stopwatch.GetTimestamp() - start);
                allocAfter = GC.GetTotalAllocatedBytes(false);
                Interlocked.Add(ref _bn2AllocBytes, allocAfter - allocBefore);

                allocBefore = GC.GetTotalAllocatedBytes(false);
                start = Stopwatch.GetTimestamp();
                var added = TensorMath.Add(null, bn2Out, input);
                Interlocked.Add(ref _addTicks, Stopwatch.GetTimestamp() - start);
                allocAfter = GC.GetTotalAllocatedBytes(false);
                Interlocked.Add(ref _addAllocBytes, allocAfter - allocBefore);

                allocBefore = GC.GetTotalAllocatedBytes(false);
                start = Stopwatch.GetTimestamp();
                var relu2 = TensorMath.ReLU(null, added);
                Interlocked.Add(ref _relu2Ticks, Stopwatch.GetTimestamp() - start);
                allocAfter = GC.GetTotalAllocatedBytes(false);
                Interlocked.Add(ref _relu2AllocBytes, allocAfter - allocBefore);

                return relu2;
            }

            allocBefore = GC.GetTotalAllocatedBytes(false);
            start = Stopwatch.GetTimestamp();
            var tOut1 = _linear1.Forward(graph, input);
            Interlocked.Add(ref _linear1Ticks, Stopwatch.GetTimestamp() - start);
            allocAfter = GC.GetTotalAllocatedBytes(false);
            Interlocked.Add(ref _linear1AllocBytes, allocAfter - allocBefore);

            allocBefore = GC.GetTotalAllocatedBytes(false);
            start = Stopwatch.GetTimestamp();
            var tBn1 = _bn1.Forward(graph, tOut1);
            Interlocked.Add(ref _bn1Ticks, Stopwatch.GetTimestamp() - start);
            allocAfter = GC.GetTotalAllocatedBytes(false);
            Interlocked.Add(ref _bn1AllocBytes, allocAfter - allocBefore);

            allocBefore = GC.GetTotalAllocatedBytes(false);
            start = Stopwatch.GetTimestamp();
            var tA1 = TensorMath.ReLU(graph, tBn1);
            Interlocked.Add(ref _relu1Ticks, Stopwatch.GetTimestamp() - start);
            allocAfter = GC.GetTotalAllocatedBytes(false);
            Interlocked.Add(ref _relu1AllocBytes, allocAfter - allocBefore);

            allocBefore = GC.GetTotalAllocatedBytes(false);
            start = Stopwatch.GetTimestamp();
            var tOut2 = _linear2.Forward(graph, tA1);
            Interlocked.Add(ref _linear2Ticks, Stopwatch.GetTimestamp() - start);
            allocAfter = GC.GetTotalAllocatedBytes(false);
            Interlocked.Add(ref _linear2AllocBytes, allocAfter - allocBefore);

            allocBefore = GC.GetTotalAllocatedBytes(false);
            start = Stopwatch.GetTimestamp();
            var tBn2 = _bn2.Forward(graph, tOut2);
            Interlocked.Add(ref _bn2Ticks, Stopwatch.GetTimestamp() - start);
            allocAfter = GC.GetTotalAllocatedBytes(false);
            Interlocked.Add(ref _bn2AllocBytes, allocAfter - allocBefore);

            allocBefore = GC.GetTotalAllocatedBytes(false);
            start = Stopwatch.GetTimestamp();
            var tAdded = TensorMath.Add(graph, tBn2, input);
            Interlocked.Add(ref _addTicks, Stopwatch.GetTimestamp() - start);
            allocAfter = GC.GetTotalAllocatedBytes(false);
            Interlocked.Add(ref _addAllocBytes, allocAfter - allocBefore);

            allocBefore = GC.GetTotalAllocatedBytes(false);
            start = Stopwatch.GetTimestamp();
            var tRelu2 = TensorMath.ReLU(graph, tAdded);
            Interlocked.Add(ref _relu2Ticks, Stopwatch.GetTimestamp() - start);
            allocAfter = GC.GetTotalAllocatedBytes(false);
            Interlocked.Add(ref _relu2AllocBytes, allocAfter - allocBefore);

            return tRelu2;
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            foreach (var p in _linear1.Parameters())
            {
                yield return p;
            }

            foreach (var p in _bn1.Parameters())
            {
                yield return p;
            }

            foreach (var p in _linear2.Parameters())
            {
                yield return p;
            }

            foreach (var p in _bn2.Parameters())
            {
                yield return p;
            }
        }

        public void Save(BinaryWriter bw)
        {
            _linear1.Save(bw);
            _bn1.Save(bw);
            _linear2.Save(bw);
            _bn2.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            _linear1.Load(br);
            _bn1.Load(br);
            _linear2.Load(br);
            _bn2.Load(br);
        }

        public void Dispose()
        {
            _linear1?.Dispose();
            _bn1?.Dispose();
            _linear2?.Dispose();
            _bn2?.Dispose();
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
