using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Core
{
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct TapeOp
    {
        public readonly OpCode Code;
        public readonly AutogradNode Output;
        public readonly AutogradNode A;
        public readonly AutogradNode B;
        public readonly object Context; // Przechowuje hiperparametry (np. wymiary CNN, tablice)

        public TapeOp(OpCode code, AutogradNode output, AutogradNode a, AutogradNode b = null, object context = null)
        {
            Code = code;
            Output = output;
            A = a;
            B = b;
            Context = context;
        }
    }
}