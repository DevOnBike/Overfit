// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Autograd
{
    /// <summary>
    /// Presents several input-column-disjoint <see cref="IDequantRowSource"/>s as one source whose rows
    /// are the per-part rows concatenated along the input (contraction) dimension; all parts share the
    /// output-row count.
    ///
    /// This is the zero-copy bridge from the GGUF loader's <b>per-head</b> output projection
    /// (<c>DecodeWeight[] Wo</c>, each <c>[dModel, headDim]</c> — output rows dModel, headDim inputs) to
    /// the combined <c>[dModel, nHeads·headDim]</c> tensor a
    /// <see cref="DevOnBike.Overfit.DeepLearning.TrainableLlamaBlock"/> expects for O. For output row
    /// <c>o</c>, each part decodes its own headDim-wide slice of the destination.
    /// </summary>
    public sealed class ConcatColsDequantSource : IDequantRowSource
    {
        private readonly IDequantRowSource[] _parts;
        private readonly int[] _colStart; // prefix sums of part InputSize; length parts+1

        public ConcatColsDequantSource(IReadOnlyList<IDequantRowSource> parts)
        {
            if (parts is null || parts.Count == 0)
            {
                throw new ArgumentException("ConcatColsDequantSource needs at least one part.", nameof(parts));
            }

            _parts = new IDequantRowSource[parts.Count];
            _colStart = new int[parts.Count + 1];
            OutputSize = parts[0].OutputSize;

            var offset = 0;
            for (var i = 0; i < parts.Count; i++)
            {
                var p = parts[i];
                if (p.OutputSize != OutputSize)
                {
                    throw new ArgumentException(
                        $"ConcatColsDequantSource: part {i} has OutputSize {p.OutputSize}, expected {OutputSize} (all parts must share the output-row count).");
                }
                _parts[i] = p;
                _colStart[i] = offset;
                offset += p.InputSize;
            }
            _colStart[parts.Count] = offset;
            InputSize = offset;
        }

        public int InputSize { get; }
        public int OutputSize { get; }

        public void DecodeRow(int row, Span<float> dst)
        {
            for (var i = 0; i < _parts.Length; i++)
            {
                _parts[i].DecodeRow(row, dst.Slice(_colStart[i], _parts[i].InputSize));
            }
        }
    }
}
