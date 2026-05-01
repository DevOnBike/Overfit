// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Autograd
{
    /// <summary>
    /// Represents a recorded operation in the computation tape.
    /// Sequential layout ensures predictable memory access during the backward pass.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct TapeOp
    {
        public readonly OpCode Code;

        public readonly AutogradNode Output;
        public readonly AutogradNode A;
        public readonly AutogradNode B;

        // Inline integer fields to avoid heap allocations for operation-specific parameters.
        public readonly int I0;
        public readonly int I1;
        public readonly int I2;
        public readonly int I3;
        public readonly int I4;

        // Fixed-size node context for hot-path operations.
        // This avoids allocating AutogradNode[] for Linear, BatchNorm1D,
        // SoftmaxCrossEntropy and similar small-context ops.
        public readonly AutogradNode C0;
        public readonly AutogradNode C1;
        public readonly AutogradNode C2;
        public readonly AutogradNode C3;
        public readonly AutogradNode C4;
        public readonly int ContextCount;

        // Variable-length fallback for ops that genuinely need array context,
        // for example StackTimesteps / FusedLSTMStep.
        public readonly AutogradNode[] NodeContext;

        // Integer array context — used for Embedding (token ids).
        // Kept separate from NodeContext to avoid boxing/casting.
        public readonly int[] IntData;

        public TapeOp(
            OpCode code,
            AutogradNode output,
            AutogradNode a,
            AutogradNode b = null,
            int i0 = 0,
            int i1 = 0,
            int i2 = 0,
            int i3 = 0,
            int i4 = 0,
            AutogradNode c0 = null,
            AutogradNode c1 = null,
            AutogradNode c2 = null,
            AutogradNode c3 = null,
            AutogradNode c4 = null,
            int contextCount = 0,
            AutogradNode[] nodeContext = null,
            int[] intData = null)
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

            C0 = c0;
            C1 = c1;
            C2 = c2;
            C3 = c3;
            C4 = c4;
            ContextCount = contextCount;

            NodeContext = nodeContext;
            IntData    = intData;
        }
    }
}