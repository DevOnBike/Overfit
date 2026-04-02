using System;
using System.Threading;

namespace DevOnBike.Overfit.Core
{
    /// <summary>
    /// Naczelne Dowództwo - zarządza Taśmą (Tape) operacji do Autogradu.
    /// Klasa jest sealed dla lepszej wydajności JIT i bezpieczeństwa architektury.
    /// </summary>
    public sealed class ComputationGraph
    {
        private const int InitialCapacity = 4096;
        private TapeOp[] _tape = new TapeOp[InitialCapacity];
        private int _opCount = 0;

        // Gwarantuje poprawną inicjalizację na każdym nowym wątku ThreadPool.
        private static readonly ThreadLocal<ComputationGraph> _active = new(() => new ComputationGraph());

        public static ComputationGraph Active
        {
            get => _active.Value;
            set => _active.Value = value;
        }

        /// <summary>
        /// Flaga pozwalająca wyłączyć nagrywanie operacji (np. podczas inferencji w UI 
        /// lub obliczania celu Bellmana w RL), co zapobiega błędom ObjectDisposedException.
        /// </summary>
        public bool IsRecording { get; set; } = true;

        public void Record(
            OpCode code,
            AutogradNode output,
            AutogradNode a, AutogradNode b = null,
        int i0 = 0, int i1 = 0, int i2 = 0, int i3 = 0, int i4 = 0,
        AutogradNode[] nodeContext = null)
        {
            if (!IsRecording) return;

            if (_opCount >= _tape.Length)
            {
                Array.Resize(ref _tape, _tape.Length * 2);
            }

            // Tworzymy strukturę na stosie i kopiujemy do tablicy (zero alokacji na stercie)
            _tape[_opCount++] = new TapeOp(code, output, a, b, i0, i1, i2, i3, i4, nodeContext);
        }

        public void Backward(AutogradNode lossNode)
        {
            // Jeśli RequiresGrad=false, macierz Grad jest null - przerywamy bezpiecznie.
            if (lossNode?.Grad == null)
            {
                return;
            }

            // Czyścimy gradienty wszystkich węzłów na taśmie, aby uniknąć ich akumulacji 
            // między różnymi krokami treningowymi.
            for (var i = 0; i < _opCount; i++)
            {
                ref readonly var op = ref _tape[i];

                op.Output?.Grad?.Clear();
                op.A?.Grad?.Clear();
                op.B?.Grad?.Clear();
            }

            // Inicjalizacja gradientu wyjściowego (Seed)
            lossNode.Grad.AsSpan().Fill(1f);

            // Reverse Pass - wykonujemy operacje w kolejności odwrotnej do nagrywania
            for (var i = _opCount - 1; i >= 0; i--)
            {
                ExecuteBackward(in _tape[i]);
            }
        }

        private void ExecuteBackward(in TapeOp op)
        {
            switch (op.Code)
            {
                case OpCode.Add:
                    TensorMath.AddBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.AddBias:
                    TensorMath.AddBiasBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.MatMul:
                    TensorMath.MatMulBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.ReLU:
                    TensorMath.ReluBackward(op.A, op.Output);
                    break;
                case OpCode.Dropout:
                    TensorMath.DropoutBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.MSELoss:
                    TensorMath.MSELossBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.SoftmaxCrossEntropy:
                    TensorMath.SoftmaxCrossEntropyBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.Conv2D:
                    TensorMath.Conv2DBackward(op.A, op.B, op.Output, op.I0, op.I1, op.I2, op.I3, op.I4);
                    break;
                case OpCode.MaxPool2D:
                    TensorMath.MaxPool2DBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.GlobalAveragePool2D:
                    TensorMath.GlobalAvgPool2DBackward(op.A, op.Output, op.I0, op.I1, op.I2);
                    break;
                case OpCode.BatchNorm1D:
                    TensorMath.BatchNorm1DBackward(op.A, op.Output, op.NodeContext[0], op.NodeContext[1], op.NodeContext[2], op.NodeContext[3]);
                    break;
            }
        }

        public void Reset()
        {
            // Zerujemy tylko te wpisy, które zostały zapisane, aby zwolnić referencje GC.
            // Array.Clear dla struktur readonly zeruje wszystkie pola (referencje i inty).
            if (_opCount > 0)
            {
                Array.Clear(_tape, 0, _opCount);
            }

            // Downsizing logic (pozostaje bez zmian dla ochrony przed Memory Bloat)
            if (_tape.Length > InitialCapacity * 2 && _opCount < _tape.Length / 4)
            {
                Array.Resize(ref _tape, Math.Max(InitialCapacity, _tape.Length / 2));
            }

            _opCount = 0;
        }
    }
}