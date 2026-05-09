// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Rope
{
    /// <summary>
    /// Precomputed RoPE (Rotary Position Embedding) cosine and sine tables.
    ///
    /// Layout: [maxSequenceLength × headDimension/2]
    ///   Cos[position, i] = cos(position / theta^(2i/headDim))
    ///   Sin[position, i] = sin(position / theta^(2i/headDim))
    ///
    /// Allocated once per session. Zero allocations during decode.
    ///
    /// Uses GPT-NeoX / Llama convention:
    ///   The rotation pairs are (x[i], x[i + headDim/2]) for i in [0, headDim/2).
    ///   This differs from the original RoPE paper which pairs adjacent elements.
    /// </summary>
    public sealed class RopeTable
    {
        private readonly float[] _cos;
        private readonly float[] _sin;
        private readonly int _halfDim;

        /// <summary>
        /// Creates and precomputes RoPE tables.
        /// </summary>
        /// <param name="maxSequenceLength">Maximum sequence length (context window).</param>
        /// <param name="headDimension">Attention head dimension (must be even).</param>
        /// <param name="theta">
        /// Base frequency. GPT-2: 10_000, Llama-3.2: 500_000, Phi-3: 10_000.
        /// </param>
        public RopeTable(int maxSequenceLength, int headDimension, float theta = 10_000f)
        {
            if (maxSequenceLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
            }
            if (headDimension <= 0 || headDimension % 2 != 0)
            {
                throw new ArgumentOutOfRangeException(nameof(headDimension), "headDimension must be positive and even.");
            }
            if (theta <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(theta));
            }

            MaxSequenceLength = maxSequenceLength;
            HeadDimension = headDimension;
            Theta = theta;

            _halfDim = headDimension / 2;

            _cos = new float[maxSequenceLength * _halfDim];
            _sin = new float[maxSequenceLength * _halfDim];

            Precompute();
        }

        public int MaxSequenceLength { get; }
        public int HeadDimension { get; }
        public float Theta { get; }

        /// <summary>Returns cos values for a given position: [halfDim].</summary>
        public ReadOnlySpan<float> CosAt(int position)
        {
            return _cos.AsSpan(position * _halfDim, _halfDim);
        }

        /// <summary>Returns sin values for a given position: [halfDim].</summary>
        public ReadOnlySpan<float> SinAt(int position)
        {
            return _sin.AsSpan(position * _halfDim, _halfDim);
        }

        private void Precompute()
        {
            for (var i = 0; i < _halfDim; i++)
            {
                // freq_i = 1 / (theta ^ (2i / headDim))
                var freq = 1f / MathF.Pow(Theta, 2f * i / HeadDimension);

                for (var pos = 0; pos < MaxSequenceLength; pos++)
                {
                    var angle = pos * freq;

                    _cos[pos * _halfDim + i] = MathF.Cos(angle);
                    _sin[pos * _halfDim + i] = MathF.Sin(angle);
                }
            }
        }
    }
}
