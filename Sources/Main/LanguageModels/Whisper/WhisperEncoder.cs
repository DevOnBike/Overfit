// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
    /// <summary>
    /// Whisper audio encoder (inference, plain-span): log-mel <c>[nMels, frames]</c> → two Conv1d stem layers
    /// (k=3; conv1 stride 1, conv2 stride 2) each + GELU → transpose to time-major → add the learned positional
    /// embedding → N pre-LN transformer blocks (non-causal self-attention + GELU MLP) → final LayerNorm.
    /// Output: encoder features <c>[nCtx, nAudioState]</c> (nCtx = frames/2), consumed by the decoder's
    /// cross-attention. Weights are pre-resolved once in the constructor and all working buffers are reused
    /// across calls, so repeated <see cref="Encode"/> calls (streaming / microphone) are allocation-stable.
    /// </summary>
    public sealed class WhisperEncoder
    {
        private sealed class Layer
        {
            public required float[] AttnLnW, AttnLnB, AttnQW, AttnQB, AttnKW, AttnVW, AttnVB, AttnOW, AttnOB;
            public required float[] MlpLnW, MlpLnB, Mlp0W, Mlp0B, Mlp2W, Mlp2B;
            public required int DFF;
        }

        private readonly int _nState;
        private readonly int _nHead;
        private readonly int _nLayer;
        private readonly int _nMels;

        private readonly float[] _conv1W, _conv1B, _conv2W, _conv2B, _pos, _lnPostW, _lnPostB;
        private readonly Layer[] _layers;
        private readonly int _maxDFF;

        // reusable buffers (grown on demand)
        private float[] _conv1 = Array.Empty<float>();
        private float[] _conv2 = Array.Empty<float>();
        private float[] _x = Array.Empty<float>();
        private float[] _ln = Array.Empty<float>();
        private float[] _attn = Array.Empty<float>();
        private float[] _hidden = Array.Empty<float>();
        private float[] _mq = Array.Empty<float>();
        private float[] _mk = Array.Empty<float>();
        private float[] _mv = Array.Empty<float>();
        private float[] _mAttnOut = Array.Empty<float>();
        private float[] _mScores = Array.Empty<float>();

        public WhisperEncoder(WhisperModel model)
        {
            _nState = model.Config.NAudioState;
            _nHead = model.Config.NAudioHead;
            _nLayer = model.Config.NAudioLayer;
            _nMels = model.Config.NMels;

            float[] T(string name) => model.Tensors.TryGetValue(name, out var t)
                ? t.Data
                : throw new KeyNotFoundException($"Whisper tensor '{name}' not found in the model.");

            _conv1W = T("encoder.conv1.weight");
            _conv1B = T("encoder.conv1.bias");
            _conv2W = T("encoder.conv2.weight");
            _conv2B = T("encoder.conv2.bias");
            _pos = T("encoder.positional_embedding");
            _lnPostW = T("encoder.ln_post.weight");
            _lnPostB = T("encoder.ln_post.bias");

            _layers = new Layer[_nLayer];
            for (var b = 0; b < _nLayer; b++)
            {
                var p = $"encoder.blocks.{b}.";
                var mlp0 = T(p + "mlp.0.weight");
                _layers[b] = new Layer
                {
                    AttnLnW = T(p + "attn_ln.weight"),
                    AttnLnB = T(p + "attn_ln.bias"),
                    AttnQW = T(p + "attn.query.weight"),
                    AttnQB = T(p + "attn.query.bias"),
                    AttnKW = T(p + "attn.key.weight"),
                    AttnVW = T(p + "attn.value.weight"),
                    AttnVB = T(p + "attn.value.bias"),
                    AttnOW = T(p + "attn.out.weight"),
                    AttnOB = T(p + "attn.out.bias"),
                    MlpLnW = T(p + "mlp_ln.weight"),
                    MlpLnB = T(p + "mlp_ln.bias"),
                    Mlp0W = mlp0,
                    Mlp0B = T(p + "mlp.0.bias"),
                    Mlp2W = T(p + "mlp.2.weight"),
                    Mlp2B = T(p + "mlp.2.bias"),
                    DFF = model.Tensors[p + "mlp.0.weight"].Shape[0],
                };
                if (_layers[b].DFF > _maxDFF)
                {
                    _maxDFF = _layers[b].DFF;
                }
            }
        }

        /// <summary>Runs the encoder on a log-mel spectrogram <c>[nMels × frames]</c> (channel/mel-major).
        /// Returns encoder features <c>[nCtx × nAudioState]</c> (time-major); <paramref name="nCtx"/> = frames/2.
        /// The returned array is a reused buffer — consume it before the next <see cref="Encode"/> call.</summary>
        public float[] Encode(ReadOnlySpan<float> logMel, int frames, out int nCtx)
        {
            nCtx = (frames + 2 * 1 - 3) / 2 + 1;

            _conv1 = Ensure(_conv1, _nState * frames);
            _conv2 = Ensure(_conv2, _nState * nCtx);
            _x = Ensure(_x, nCtx * _nState);
            _ln = Ensure(_ln, nCtx * _nState);
            _attn = Ensure(_attn, nCtx * _nState);
            _hidden = Ensure(_hidden, nCtx * _maxDFF);
            _mq = Ensure(_mq, nCtx * _nState);
            _mk = Ensure(_mk, nCtx * _nState);
            _mv = Ensure(_mv, nCtx * _nState);
            _mAttnOut = Ensure(_mAttnOut, nCtx * _nState);
            _mScores = Ensure(_mScores, nCtx);

            // ── Conv1d stem ──
            var conv1 = _conv1.AsSpan(0, _nState * frames);
            WhisperKernels.Conv1d(logMel, _conv1W, _conv1B, conv1, _nMels, frames, _nState, 3, 1, 1, frames);
            WhisperKernels.GeluInPlace(conv1);

            var conv2 = _conv2.AsSpan(0, _nState * nCtx);
            WhisperKernels.Conv1d(conv1, _conv2W, _conv2B, conv2, _nState, frames, _nState, 3, 2, 1, nCtx);
            WhisperKernels.GeluInPlace(conv2);

            // ── transpose [nState, nCtx] → [nCtx, nState] and add positional embedding ──
            var x = _x.AsSpan(0, nCtx * _nState);
            for (var t = 0; t < nCtx; t++)
            {
                for (var c = 0; c < _nState; c++)
                {
                    x[t * _nState + c] = conv2[c * nCtx + t] + _pos[t * _nState + c];
                }
            }

            // ── transformer blocks (pre-LN) ──
            var ln = _ln.AsSpan(0, nCtx * _nState);
            var attn = _attn.AsSpan(0, nCtx * _nState);
            for (var bi = 0; bi < _nLayer; bi++)
            {
                var l = _layers[bi];

                WhisperKernels.LayerNorm(x, l.AttnLnW, l.AttnLnB, ln, nCtx, _nState);
                WhisperKernels.MultiHeadAttention(
                    ln, nCtx, ln, nCtx, _nState, _nHead,
                    l.AttnQW, l.AttnQB, l.AttnKW, l.AttnVW, l.AttnVB, l.AttnOW, l.AttnOB,
                    attn, causal: false,
                    _mq.AsSpan(0, nCtx * _nState), _mk.AsSpan(0, nCtx * _nState), _mv.AsSpan(0, nCtx * _nState),
                    _mAttnOut.AsSpan(0, nCtx * _nState), _mScores.AsSpan(0, nCtx));
                TensorPrimitives.Add(x, attn, x);

                WhisperKernels.LayerNorm(x, l.MlpLnW, l.MlpLnB, ln, nCtx, _nState);
                var hidden = _hidden.AsSpan(0, nCtx * l.DFF);
                WhisperKernels.Linear(ln, l.Mlp0W, l.Mlp0B, hidden, nCtx, _nState, l.DFF);
                WhisperKernels.GeluInPlace(hidden);
                WhisperKernels.Linear(hidden, l.Mlp2W, l.Mlp2B, attn, nCtx, l.DFF, _nState);
                TensorPrimitives.Add(x, attn, x);
            }

            WhisperKernels.LayerNorm(x, _lnPostW, _lnPostB, x, nCtx, _nState);
            return _x;
        }

#pragma warning disable OVERFIT001 // Grows a REUSED encoder scratch buffer only when too small (amortized one-time); steady-state encode reuses the pre-allocated buffers.
        private static float[] Ensure(float[] buffer, int needed)
            => buffer.Length >= needed ? buffer : new float[needed];
#pragma warning restore OVERFIT001
    }
}
