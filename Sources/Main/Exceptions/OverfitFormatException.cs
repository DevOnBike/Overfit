// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Exceptions
{
    /// <summary>
    /// A model or data file is malformed, truncated, or internally inconsistent — GGUF / safetensors / ONNX
    /// parsing, a checkpoint that doesn't match, a bad header or magic. The Overfit-domain replacement for
    /// <see cref="System.IO.InvalidDataException"/> at the runtime's parsing sites.
    /// </summary>
    public class OverfitFormatException : OverfitException
    {
        public OverfitFormatException()
        {
        }

        public OverfitFormatException(string message)
            : base(message)
        {
        }

        public OverfitFormatException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
