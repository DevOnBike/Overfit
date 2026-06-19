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
        private readonly RopeScaling? _scaling;
        private readonly float[]? _freqFactors;   // Phi-3 longrope per-dim divisor [halfDim]; null = none
        private readonly float _attnFactor;       // longrope mscale on cos/sin; 1 = none

        /// <summary>
        /// Creates and precomputes RoPE tables.
        /// </summary>
        /// <param name="maxSequenceLength">Maximum sequence length (context window).</param>
        /// <param name="headDimension">Attention head dimension (must be even).</param>
        /// <param name="theta">
        /// Base frequency. GPT-2: 10_000, Llama-3.2: 500_000, Phi-3: 10_000.
        /// </param>
        /// <param name="scaling">
        /// Optional Llama-3 "llama3" frequency scaling for long context; null = plain RoPE.
        /// </param>
        /// <param name="splitHalf">
        /// When true, uses the split-half rotation layout (rotate-half over the two contiguous halves of
        /// each head) instead of the adjacent-pair layout. Default false.
        /// </param>
        public RopeTable(int maxSequenceLength, int headDimension, float theta = 10_000f, RopeScaling? scaling = null, bool splitHalf = false, float[]? freqFactors = null, float attnFactor = 1f)
        {
            SplitHalf = splitHalf;
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSequenceLength);
            if (headDimension <= 0 || headDimension % 2 != 0)
            {
                throw new ArgumentOutOfRangeException(nameof(headDimension), "headDimension must be positive and even.");
            }
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(theta, 0f);
            if (freqFactors is not null && freqFactors.Length != headDimension / 2)
            {
                throw new ArgumentException(
                    $"freqFactors length ({freqFactors.Length}) must equal head_dim/2 ({headDimension / 2}).", nameof(freqFactors));
            }

            MaxSequenceLength = maxSequenceLength;
            HeadDimension = headDimension;
            Theta = theta;
            _scaling = scaling;
            _freqFactors = freqFactors;
            _attnFactor = attnFactor;

            _halfDim = headDimension / 2;

            _cos = new float[maxSequenceLength * _halfDim];
            _sin = new float[maxSequenceLength * _halfDim];

            Precompute();
        }

        public int MaxSequenceLength
        {
            get;
        }
        public int HeadDimension
        {
            get;
        }
        public float Theta
        {
            get;
        }

        /// <summary>
        /// When true, RoPE pairs split-half dims <c>(x[i], x[i+d/2])</c> (HF rotate_half / NEOX); when
        /// false, adjacent dims <c>(x[2i], x[2i+1])</c>. See <c>GPT1Config.RopeSplitHalf</c>.
        /// </summary>
        public bool SplitHalf
        {
            get;
        }

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
                // freq_i = 1 / (theta ^ (2i / headDim)), optionally llama3-rescaled.
                var freq = 1f / MathF.Pow(Theta, 2f * i / HeadDimension);
                if (_scaling is not null)
                {
                    freq = _scaling.Apply(freq);
                }
                // Phi-3 longrope: divide each dim's base frequency by its per-dim factor.
                if (_freqFactors is not null)
                {
                    freq /= _freqFactors[i];
                }

                for (var pos = 0; pos < MaxSequenceLength; pos++)
                {
                    var angle = pos * freq;

                    _cos[pos * _halfDim + i] = _attnFactor * MathF.Cos(angle);
                    _sin[pos * _halfDim + i] = _attnFactor * MathF.Sin(angle);
                }
            }
        }
    }
}
