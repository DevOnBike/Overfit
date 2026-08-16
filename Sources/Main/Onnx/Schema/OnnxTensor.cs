// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Onnx.Schema
{
    /// <summary>
    /// Tensor data — used for both model weight initializers and inline constants.
    /// After external data resolution, RawData always contains the decoded bytes.
    /// </summary>
    public sealed class OnnxTensor
    {
        public string Name { get; init; } = "";
        public OnnxDataType DataType
        {
            get; init;
        }
        public long[] Dims { get; init; } = [];

        /// <summary>Raw little-endian bytes. Populated after ResolveExternalData.</summary>
        public byte[] RawData { get; init; } = [];

        /// <summary>Float data when stored unpacked in the protobuf (alternative to RawData).</summary>
        public float[]? FloatData
        {
            get; init;
        }

        /// <summary>Int64 data when stored unpacked.</summary>
        public long[]? Int64Data
        {
            get; init;
        }

        /// <summary>
        /// Non-null when data lives in a separate .data file (PyTorch ≥ 2.x default).
        /// Resolved to RawData by OnnxImporter before operator mapping begins.
        /// </summary>
        public OnnxExternalDataInfo? ExternalData
        {
            get; init;
        }

        /// <summary>
        /// Elements the declared shape describes.
        ///
        /// <para><b>Checked, and the reason is not the crash.</b> The dimensions come from a file. Multiplied
        /// unchecked they wrap, and a wrapped product can be small, positive and plausible — so it passes a
        /// later shape comparison while the real on-disk layout disagrees, and the model loads with silently
        /// wrong weights and no exception at all. That is strictly worse than a refusal, because the only
        /// symptom is output that is merely a bit worse than it should be.</para>
        ///
        /// <para>The same defect existed in <c>GgufTensorInfo.ElementCount</c> and in the graph importer's
        /// size computation; all three were fixed together.</para>
        /// </summary>
        /// <exception cref="OverfitFormatException">
        /// A dimension is negative, or the product does not fit an <see cref="int"/>.
        /// </exception>
        public int ElementCount
        {
            get
            {
                long count = 1;

                foreach (var d in Dims)
                {
                    if (d < 0)
                    {
                        throw new OverfitFormatException(
                            $"Tensor '{Name}' declares a negative dimension ({d}).");
                    }

                    count *= d;

                    if (count > int.MaxValue)
                    {
                        throw new OverfitFormatException(
                            $"Tensor '{Name}' declares {count} or more elements, which exceeds the "
                            + $"{int.MaxValue} this runtime can address.");
                    }
                }

                return (int)count;
            }
        }
    }
}
