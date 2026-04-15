// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Autograd
{
    /// <summary>
    ///     Represents a recorded operation in the computation tape.
    ///     Sequential layout ensures predictable memory access during the backward pass.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct TapeOp
    {
        public readonly OpCode Code;
        public readonly AutogradNode Output;
        public readonly AutogradNode A;
        public readonly AutogradNode B;

        // Inline integer fields to avoid 'new int[]' for operation-specific parameters (Conv/Stride).
        public readonly int I0, I1, I2, I3, I4;

        // Multi-node context for complex layers (e.g., BatchNorm storing mean/var).
        public readonly AutogradNode[] NodeContext;

        public TapeOp(OpCode code, AutogradNode output, AutogradNode a, AutogradNode b = null,
            int i0 = 0, int i1 = 0, int i2 = 0, int i3 = 0, int i4 = 0,
            AutogradNode[] nodeContext = null)
        {
            Code = code;
            Output = output;
            A = a;
            B = b;
            I0 = i0;
            I1 = i1;
            I2 = i2;
            I3 = i3;
            I4 = i4;
            NodeContext = nodeContext;
        }
    }
}