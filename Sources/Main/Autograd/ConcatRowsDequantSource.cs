// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Autograd
{
    /// <summary>
    /// Presents several output-row-disjoint <see cref="IDequantRowSource"/>s as one stacked source —
    /// rows are concatenated along the output dimension, the input (contraction) dimension is shared.
    ///
    /// This is the zero-copy bridge from the GGUF loader's <b>per-head</b> attention weights
    /// (<c>DecodeWeight[] Wq</c>, each <c>[headDim, dModel]</c>) to the combined
    /// <c>[nHeads·headDim, dModel]</c> tensor a <see cref="DevOnBike.Overfit.DeepLearning.TrainableLlamaBlock"/>
    /// expects for Q / K / V — without repacking the quantized bytes. Output row <c>r</c> maps to the
    /// part holding it and that part's local row.
    /// </summary>
    public sealed class ConcatRowsDequantSource : IDequantRowSource
    {
        private readonly IDequantRowSource[] _parts;
        private readonly int[] _rowStart; // prefix sums of part OutputSize; length parts+1

        public ConcatRowsDequantSource(IReadOnlyList<IDequantRowSource> parts)
        {
            if (parts is null || parts.Count == 0)
            {
                throw new ArgumentException("ConcatRowsDequantSource needs at least one part.", nameof(parts));
            }

            _parts = new IDequantRowSource[parts.Count];
            _rowStart = new int[parts.Count + 1];
            InputSize = parts[0].InputSize;

            var offset = 0;
            for (var i = 0; i < parts.Count; i++)
            {
                var p = parts[i];
                if (p.InputSize != InputSize)
                {
                    throw new ArgumentException(
                        $"ConcatRowsDequantSource: part {i} has InputSize {p.InputSize}, expected {InputSize} (all parts must share the contraction dim).");
                }
                _parts[i] = p;
                _rowStart[i] = offset;
                offset += p.OutputSize;
            }
            _rowStart[parts.Count] = offset;
            OutputSize = offset;
        }

        public int InputSize
        {
            get;
        }
        public int OutputSize
        {
            get;
        }

        public void DecodeRow(int row, Span<float> dst)
        {
            for (var i = 0; i < _parts.Length; i++)
            {
                if (row < _rowStart[i + 1])
                {
                    _parts[i].DecodeRow(row - _rowStart[i], dst);
                    return;
                }
            }
            throw new ArgumentOutOfRangeException(nameof(row), row, $"row out of range [0, {OutputSize}).");
        }
    }
}
