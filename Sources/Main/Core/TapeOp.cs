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

        // Inline fields - eliminują "new int[]" w Conv2D i GAP
        public readonly int I0, I1, I2, I3, I4;
        
        // Kontekst dla operacji wymagających wielu węzłów (np. BatchNorm)
        public readonly AutogradNode[] NodeContext;

        public TapeOp(OpCode code, AutogradNode output, AutogradNode a, AutogradNode b = null, 
            int i0 = 0, int i1 = 0, int i2 = 0, int i3 = 0, int i4 = 0, 
            AutogradNode[] nodeContext = null)
        {
            Code = code;
            Output = output;
            A = a;
            B = b;
            I0 = i0; I1 = i1; I2 = i2; I3 = i3; I4 = i4;
            NodeContext = nodeContext;
        }
    }
}