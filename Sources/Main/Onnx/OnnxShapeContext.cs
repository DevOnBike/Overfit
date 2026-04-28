// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Onnx
{
    /// <summary>
    /// Tracks concrete tensor shapes propagating through the ONNX graph during import.
    /// Operators read the input shape they need, compute the output shape, and write it
    /// back so subsequent operators can use it.
    /// </summary>
    internal sealed class OnnxShapeContext
    {
        private readonly Dictionary<string, int[]> _shapes = new();

        public void SetShape(string tensorName, int[] shape)
        {
            _shapes[tensorName] = shape;
        }

        public int[]? GetShape(string tensorName)
        {
            return _shapes.TryGetValue(tensorName, out var shape) ? shape : null;
        }
    }
}
