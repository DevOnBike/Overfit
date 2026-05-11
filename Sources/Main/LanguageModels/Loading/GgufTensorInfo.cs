// Copyright (c) 2026 DevOnBike. AGPLv3.

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// Descriptor of a single tensor in a GGUF file.
    /// Note: dimensions are in GGUF order (fastest-stride first).
    /// To get HuggingFace shape semantics, reverse the dim order.
    /// </summary>
    public sealed class GgufTensorInfo
    {
        public GgufTensorInfo(
            string name,
            ulong[] dims,
            GgmlType type,
            ulong offset)
        {
            Name = name;
            Dims = dims;
            Type = type;
            Offset = offset;
        }

        /// <summary>Tensor name (e.g. "blk.0.attn_q.weight").</summary>
        public string Name { get; }

        /// <summary>Dimensions in GGUF order (fastest-stride first).</summary>
        public ulong[] Dims { get; }

        /// <summary>GGML data type (F32, F16, etc.).</summary>
        public GgmlType Type { get; }

        /// <summary>Byte offset from the start of the data section.</summary>
        public ulong Offset { get; }

        /// <summary>Total number of elements (product of all dims).</summary>
        public long ElementCount
        {
            get
            {
                var n = 1L;
                for (var i = 0; i < Dims.Length; i++)
                {
                    n *= (long)Dims[i];
                }
                return n;
            }
        }
    }
}
