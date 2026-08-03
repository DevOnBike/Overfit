// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Exceptions;

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
        public string Name
        {
            get;
        }

        /// <summary>Dimensions in GGUF order (fastest-stride first).</summary>
        public ulong[] Dims
        {
            get;
        }

        /// <summary>GGML data type (F32, F16, etc.).</summary>
        public GgmlType Type
        {
            get;
        }

        /// <summary>Byte offset from the start of the data section.</summary>
        public ulong Offset
        {
            get;
        }

        /// <summary>
        /// Total number of elements (product of all dims).
        ///
        /// <para><b>Checked, and the crash is not the reason.</b> The dimensions are <see cref="ulong"/>
        /// read straight from the file. Cast to <see cref="long"/> and multiplied unchecked they wrap, and
        /// the dangerous outcome is not an exception but a product that is small, positive and plausible: it
        /// can <b>match</b> the loader's expected-shape check while the real on-disk layout disagrees, so
        /// the model loads with silently wrong weights and produces output that is merely worse. A refusal
        /// is strictly better than that.</para>
        ///
        /// <para><c>SafetensorsReader</c> guards the same class of arithmetic explicitly in
        /// <c>RequireTensorsFitInTheDataBlock</c>; this is the sibling that did not. The same product was
        /// unchecked in <c>OnnxTensor.ElementCount</c> and the ONNX graph importer, and all three were fixed
        /// together.</para>
        /// </summary>
        /// <exception cref="OverfitFormatException">
        /// A dimension exceeds <see cref="long.MaxValue"/>, or the product overflows.
        /// </exception>
        public long ElementCount
        {
            get
            {
                var n = 1L;

                for (var i = 0; i < Dims.Length; i++)
                {
                    if (Dims[i] > long.MaxValue)
                    {
                        throw new OverfitFormatException(
                            $"Tensor '{Name}' declares dimension {i} as {Dims[i]}, which is not a usable "
                            + "element count.");
                    }

                    var dim = (long)Dims[i];

                    if (dim != 0 && n > long.MaxValue / dim)
                    {
                        throw new OverfitFormatException(
                            $"Tensor '{Name}' declares dimensions whose product overflows: "
                            + $"{string.Join(" x ", Dims)}.");
                    }

                    n *= dim;
                }

                return n;
            }
        }
    }
}
