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
    /// cross-attention. Weights are taken from a loaded <see cref="WhisperModel"/> by their whisper.cpp names.
    /// </summary>
    public sealed class WhisperEncoder
    {
        private readonly WhisperModel _m;
        private readonly int _nState;
        private readonly int _nHead;
        private readonly int _nLayer;
        private readonly int _nMels;

        public WhisperEncoder(WhisperModel model)
        {
            _m = model;
            _nState = model.Config.NAudioState;
            _nHead = model.Config.NAudioHead;
            _nLayer = model.Config.NAudioLayer;
            _nMels = model.Config.NMels;
        }

        /// <summary>Runs the encoder on a log-mel spectrogram <c>[nMels × frames]</c> (channel/mel-major).
        /// Returns encoder features <c>[nCtx × nAudioState]</c> (time-major); <paramref name="nCtx"/> = frames/2.</summary>
        public float[] Encode(ReadOnlySpan<float> logMel, int frames, out int nCtx)
        {
            // ── Conv1d stem ──
            var conv1 = new float[_nState * frames];
            WhisperKernels.Conv1d(logMel, T("encoder.conv1.weight"), T("encoder.conv1.bias"),
                conv1, _nMels, frames, _nState, kSize: 3, stride: 1, pad: 1, tOut: frames);
            WhisperKernels.GeluInPlace(conv1);

            nCtx = (frames + 2 * 1 - 3) / 2 + 1;
            var conv2 = new float[_nState * nCtx];
            WhisperKernels.Conv1d(conv1, T("encoder.conv2.weight"), T("encoder.conv2.bias"),
                conv2, _nState, frames, _nState, kSize: 3, stride: 2, pad: 1, tOut: nCtx);
            WhisperKernels.GeluInPlace(conv2);

            // ── transpose [nState, nCtx] → [nCtx, nState] and add positional embedding ──
            var x = new float[nCtx * _nState];
            var pos = T("encoder.positional_embedding");
            for (var t = 0; t < nCtx; t++)
            {
                for (var c = 0; c < _nState; c++)
                {
                    x[t * _nState + c] = conv2[c * nCtx + t] + pos[t * _nState + c];
                }
            }

            // ── transformer blocks (pre-LN) ──
            var ln = new float[nCtx * _nState];
            var attn = new float[nCtx * _nState];
            for (var b = 0; b < _nLayer; b++)
            {
                var p = $"encoder.blocks.{b}.";

                // self-attention
                WhisperKernels.LayerNorm(x, T(p + "attn_ln.weight"), T(p + "attn_ln.bias"), ln, nCtx, _nState);
                WhisperKernels.MultiHeadAttention(
                    ln, nCtx, ln, nCtx, _nState, _nHead,
                    T(p + "attn.query.weight"), T(p + "attn.query.bias"),
                    T(p + "attn.key.weight"),
                    T(p + "attn.value.weight"), T(p + "attn.value.bias"),
                    T(p + "attn.out.weight"), T(p + "attn.out.bias"),
                    attn, causal: false);
                TensorPrimitives.Add(x, attn, x);

                // MLP
                WhisperKernels.LayerNorm(x, T(p + "mlp_ln.weight"), T(p + "mlp_ln.bias"), ln, nCtx, _nState);
                var w0 = _m.Tensors[p + "mlp.0.weight"];
                var dFF = w0.Shape[0];
                var hidden = new float[nCtx * dFF];
                WhisperKernels.Linear(ln, w0.Data, T(p + "mlp.0.bias"), hidden, nCtx, _nState, dFF);
                WhisperKernels.GeluInPlace(hidden);
                WhisperKernels.Linear(hidden, T(p + "mlp.2.weight"), T(p + "mlp.2.bias"), attn, nCtx, dFF, _nState);
                TensorPrimitives.Add(x, attn, x);
            }

            WhisperKernels.LayerNorm(x, T("encoder.ln_post.weight"), T("encoder.ln_post.bias"), x, nCtx, _nState);
            return x;
        }

        private float[] T(string name)
            => _m.Tensors.TryGetValue(name, out var t)
                ? t.Data
                : throw new KeyNotFoundException($"Whisper tensor '{name}' not found in the model.");
    }
}
