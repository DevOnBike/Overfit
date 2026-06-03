// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Autograd
{
    /// <summary>
    /// A frozen, output-major weight matrix that can dequantize one output row
    /// (<see cref="OutputSize"/> rows, each an <see cref="InputSize"/>-long contraction vector)
    /// into an F32 span on demand. Implemented by the K-quant weights (Q4_K / Q6_K) so the
    /// autograd <c>FrozenQuantizedLinear</c> op can run a training forward/backward against a
    /// 4–6-bit base that stays quantized in RAM (the QLoRA base path). The base is frozen:
    /// the op never asks for or produces a weight gradient.
    /// </summary>
    public interface IDequantRowSource
    {
        /// <summary>Contraction-dimension length (input features).</summary>
        int InputSize { get; }

        /// <summary>Number of output rows.</summary>
        int OutputSize { get; }

        /// <summary>Dequantizes output row <paramref name="row"/> into <paramref name="dst"/> (F32,
        /// length ≥ <see cref="InputSize"/>). No allocation; decodes straight from the resident
        /// quantized bytes.</summary>
        void DecodeRow(int row, System.Span<float> dst);
    }
}
