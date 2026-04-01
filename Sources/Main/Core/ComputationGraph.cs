namespace DevOnBike.Overfit.Core
{
    public class ComputationGraph
    {
        private TapeOp[] _tape = new TapeOp[4096];
        private int _opCount = 0;

        // ThreadLocal gwarantuje, że każdy wątek otrzyma swoją własną, 
        // zainicjowaną instancję grafu przy pierwszym odwołaniu.
        private static readonly ThreadLocal<ComputationGraph> _active = new ThreadLocal<ComputationGraph>(() => new ComputationGraph());

        // Publiczna właściwość, która ukrywa przed resztą kodu fakt użycia ThreadLocal.
        // Dzięki temu Twoje wywołania ComputationGraph.Active.Record(...) nadal działają!
        public static ComputationGraph Active
        {
            get => _active.Value;
            set => _active.Value = value;
        }

        // Flaga blokująca nagrywanie operacji ewaluacyjnych
        public bool IsRecording { get; set; } = true;

        public void Record(OpCode code, AutogradNode output, AutogradNode a, AutogradNode b = null, object context = null)
        {
            // TĘ LINIJKĘ MUSISZ DODAĆ! Jeśli jesteśmy w trybie ewaluacji, przerywamy nagrywanie na taśmę.
            if (!IsRecording) return;

            if (_opCount >= _tape.Length)
            {
                Array.Resize(ref _tape, _tape.Length * 2);
            }

            _tape[_opCount++] = new TapeOp(code, output, a, b, context);
        }

        public void Backward(AutogradNode lossNode)
        {
            // Przechodzimy po całej taśmie i czyścimy gradienty wszystkich węzłów.
            // Dzięki temu mamy pewność, że nie nakładamy nowych wyników na "śmieci" 
            // z poprzedniego wywołania Backward.
            for (int i = 0; i < _opCount; i++)
            {
                ref readonly var op = ref _tape[i];

                op.Output?.Grad?.Clear();
                op.A?.Grad?.Clear();
                op.B?.Grad?.Clear();
            }

            // Seedowanie gradientu wyjściowego (początek łańcucha)
            lossNode.Grad.AsSpan().Fill(1.0);

            // Właściwa faza Backward (Reverse Pass)
            for (var i = _opCount - 1; i >= 0; i--)
            {
                // Używamy 'in' dla wydajności (brak kopiowania struktury TapeOp)
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
                case OpCode.MaxPool2D:
                    TensorMath.MaxPool2DBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.Conv2D:
                    var cCtx = (int[])op.Context;
                    TensorMath.Conv2DBackward(op.A, op.B, op.Output, cCtx[0], cCtx[1], cCtx[2], cCtx[3], cCtx[4]);
                    break;
                case OpCode.GlobalAveragePool2D:
                    var gCtx = (int[])op.Context;
                    TensorMath.GlobalAvgPool2DBackward(op.A, op.Output, gCtx[0], gCtx[1], gCtx[2]);
                    break;
                case OpCode.BatchNorm1D:
                    var bnCtx = (AutogradNode[])op.Context;
                    TensorMath.BatchNorm1DBackward(op.A, op.Output, bnCtx[0], bnCtx[1], bnCtx[2], bnCtx[3]);
                    break;
            }
        }

        public void Reset()
        {
            // Oprócz wyzerowania licznika, MUSIMY wyczyścić referencje w tablicy.
            // Inaczej _tape będzie trzymał przy życiu "Węzły-Zombie" z poprzedniego kroku,
            // blokując oddanie pamięci z powrotem do ArrayPool!
            if (_opCount > 0)
            {
                Array.Clear(_tape, 0, _opCount);
            }

            _opCount = 0;
        }
    }
}