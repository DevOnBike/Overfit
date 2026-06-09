// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Exposes a contiguous <b>row range</b> <c>[start, start+count)</c> of an <see cref="IDequantRowSource"/> as a
    /// smaller weight. Lets the LM head produce logits over just a sub-vocabulary (e.g. only the audio tokens for
    /// voice cloning) without any new autograd op — <c>FrozenQuantizedLinear</c> is vocab-agnostic, so it runs
    /// unchanged against this view. Generated indices are offset by <c>start</c> back to real token ids by the caller.
    /// </summary>
    internal sealed class RangedDequantRowSource : IDequantRowSource
    {
        private readonly IDequantRowSource _inner;
        private readonly int _start;
        private readonly int _count;

        public RangedDequantRowSource(IDequantRowSource inner, int start, int count)
        {
            if (start < 0 || count <= 0 || start + count > inner.OutputSize)
            {
                throw new ArgumentOutOfRangeException(nameof(count),
                    $"Range [{start}, {start + count}) is outside the weight's {inner.OutputSize} rows.");
            }
            _inner = inner;
            _start = start;
            _count = count;
        }

        public int InputSize => _inner.InputSize;

        public int OutputSize => _count;

        public void DecodeRow(int row, Span<float> dst) => _inner.DecodeRow(_start + row, dst);
    }
}
