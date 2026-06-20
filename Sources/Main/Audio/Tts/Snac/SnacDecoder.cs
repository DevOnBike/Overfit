// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts.Snac
{
    /// <summary>
    /// The SNAC codec decoder forward pass — residual-VQ codes back to a 24 kHz waveform — wired exactly to the
    /// reference graph: <c>from_codes</c> (per level: codebook gather → 1×1 <c>out_proj</c> → repeat-interleave →
    /// sum) → depthwise input conv → 1×1 to <c>decoder_dim</c> → four <see cref="DecoderBlock"/>s (Snake →
    /// transposed-conv upsample → optional noise → 3 dilated <see cref="ResidualUnit"/>s) → Snake → output conv →
    /// <c>tanh</c>. Correctness-first plain-span (decode runs once per utterance, not a per-token hot loop).
    /// </summary>
    internal sealed class SnacDecoder
    {
        private static readonly int[] ResidualDilations = [1, 3, 9];

        private readonly SnacConfig _cfg;
        private readonly SnacWeights _w;

        public SnacDecoder(SnacConfig config, SnacWeights weights)
        {
            _cfg = config;
            _w = weights;
        }

        /// <summary>
        /// Decodes one utterance. <paramref name="codes"/> has one int array per VQ level (coarse→fine, matching
        /// <see cref="SnacConfig.VqStrides"/>); each level's length is the latent frame count divided by its
        /// stride. <paramref name="addNoise"/> enables the stochastic NoiseBlock (off → deterministic, the path the
        /// parity gate checks). Returns mono PCM in <c>[-1, 1]</c> at <see cref="SnacConfig.SampleRate"/>.
        /// </summary>
        public float[] Decode(int[][] codes, bool addNoise = false)
        {
            if (codes.Length != _cfg.CodebookCount)
            {
                throw new OverfitRuntimeException(
                    $"SNAC expects {_cfg.CodebookCount} code levels but got {codes.Length}.");
            }

            var zq = FromCodes(codes, out var frames);
            return RunDecoder(zq, frames, addNoise);
        }

        // ── ResidualVectorQuantize.from_codes: Σ_level repeat_interleave(out_proj(decode_code(codes_i))) ──
        private float[] FromCodes(int[][] codes, out int frames)
        {
            var latent = _cfg.LatentDim;
            var cbDim = _cfg.CodebookDim;
            // The stride-1 (finest) level sets the latent length; all levels upsample to it.
            frames = codes[^1].Length;
            var zq = new float[latent * frames];

            for (var i = 0; i < codes.Length; i++)
            {
                var levelCodes = codes[i];
                var ti = levelCodes.Length;
                var stride = _cfg.VqStrides[i];
                if (ti * stride != frames)
                {
                    throw new OverfitRuntimeException(
                        $"SNAC level {i} has {ti} codes; expected {frames / stride} for stride {stride}.");
                }

                // decode_code: codebook gather → [cbDim × ti]
                var zp = new float[cbDim * ti];
                SnacResidualVq.DecodeCodebook(levelCodes, _w[$"q.{i}.codebook"], zp, _cfg.CodebookSize, cbDim, ti);

                // out_proj: 1×1 conv cbDim → latent → [latent × ti]
                var proj = new float[latent * ti];
                SnacConv.Conv1d(zp, _w[$"q.{i}.out_proj.weight"], _w[$"q.{i}.out_proj.bias"], proj,
                    inC: cbDim, tIn: ti, outC: latent, kSize: 1, stride: 1, pad: 0, dilation: 1, groups: 1, tOut: ti);

                // repeat_interleave(stride) → [latent × frames], accumulate
                if (stride == 1)
                {
                    for (var j = 0; j < zq.Length; j++)
                    {
                        zq[j] += proj[j];
                    }
                }
                else
                {
                    var up = new float[latent * frames];
                    SnacResidualVq.RepeatInterleaveTime(proj, up, latent, ti, stride);
                    for (var j = 0; j < zq.Length; j++)
                    {
                        zq[j] += up[j];
                    }
                }
            }

            return zq;
        }

        // ── Decoder.model: depthwise stem → blocks → snake → out conv → tanh ──
        private float[] RunDecoder(ReadOnlySpan<float> zq, int frames, bool addNoise)
        {
            var latent = _cfg.LatentDim;

            // dec.in_dw: depthwise k7 pad3 (length preserved)
            var dw = new float[latent * frames];
            SnacConv.Conv1d(zq, _w["dec.in_dw.weight"], _w["dec.in_dw.bias"], dw,
                inC: latent, tIn: frames, outC: latent, kSize: 7, stride: 1, pad: 3, dilation: 1, groups: latent, tOut: frames);

            // dec.in_pw: 1×1 latent → decoder_dim
            var dim = _cfg.DecoderDim;
            var x = new float[dim * frames];
            SnacConv.Conv1d(dw, _w["dec.in_pw.weight"], _w["dec.in_pw.bias"], x,
                inC: latent, tIn: frames, outC: dim, kSize: 1, stride: 1, pad: 0, dilation: 1, groups: 1, tOut: frames);

            var curT = frames;
            for (var b = 0; b < _cfg.DecoderRates.Length; b++)
            {
                x = DecoderBlock(x, dim, curT, b, addNoise, out var outDim, out var outT);
                dim = outDim;
                curT = outT;
            }

            // dec.snake_out + dec.out_conv (k7 pad3) → 1 channel, then tanh
            SnacActivations.Snake1dInPlace(x, _w["dec.snake_out.alpha"], dim, curT);
            var audio = new float[curT];
            SnacConv.Conv1d(x, _w["dec.out_conv.weight"], _w["dec.out_conv.bias"], audio,
                inC: dim, tIn: curT, outC: 1, kSize: 7, stride: 1, pad: 3, dilation: 1, groups: 1, tOut: curT);
            for (var i = 0; i < audio.Length; i++)
            {
                audio[i] = MathF.Tanh(audio[i]);
            }
            return audio;
        }

        // DecoderBlock: Snake → ConvTranspose1d(k=2·stride, pad=⌈stride/2⌉, output_padding=stride%2) → [noise] → 3 ResidualUnits
        private float[] DecoderBlock(Span<float> x, int inDim, int tIn, int b, bool addNoise, out int outDim, out int outT)
        {
            outDim = inDim / 2;
            var stride = _cfg.DecoderRates[b];
            var k = 2 * stride;
            var pad = (stride + 1) / 2;          // ceil(stride/2)
            var outputPad = stride % 2;

            SnacActivations.Snake1dInPlace(x, _w[$"dec.block.{b}.snake_in.alpha"], inDim, tIn);

            outT = SnacConv.OutputLength(tIn, k, stride, pad, dilation: 1, outputPadding: outputPad);
            var y = new float[outDim * outT];
            SnacConv.ConvTranspose1d(x, _w[$"dec.block.{b}.convt.weight"], _w[$"dec.block.{b}.convt.bias"], y,
                inC: inDim, tIn: tIn, outC: outDim, kSize: k, stride: stride, pad: pad, dilation: 1, tOut: outT);

            if (addNoise)
            {
                ApplyNoise(y, _w[$"dec.block.{b}.noise.weight"], outDim, outT, b);
            }

            for (var r = 0; r < ResidualDilations.Length; r++)
            {
                y = SnacBlocks.ResidualUnit(_w, $"dec.block.{b}.res.{r}.", y, outDim, outT, ResidualDilations[r]);
            }
            return y;
        }

        // NoiseBlock: x += randn(1,T) ⊙ (1×1 conv of x). Random by design; only used when addNoise is requested.
        private static void ApplyNoise(Span<float> x, ReadOnlySpan<float> noiseWeight, int dim, int t, int seedOffset)
        {
            var h = new float[dim * t];
            SnacConv.Conv1d(x, noiseWeight, ReadOnlySpan<float>.Empty, h,
                inC: dim, tIn: t, outC: dim, kSize: 1, stride: 1, pad: 0, dilation: 1, groups: 1, tOut: t);

            // One noise sample per time step, shared across channels (matches randn((B,1,T))).
            var state = (uint)(0x9E3779B9 + seedOffset);
            for (var ti = 0; ti < t; ti++)
            {
                var n = NextGaussian(ref state);
                for (var c = 0; c < dim; c++)
                {
                    x[(c * t) + ti] += n * h[(c * t) + ti];
                }
            }
        }

        // Box–Muller from a deterministic LCG (no Math.Random; reproducible per call).
        private static float NextGaussian(ref uint state)
        {
            state = (state * 1664525u) + 1013904223u;
            var u1 = ((state >> 8) + 1u) / (float)(1 << 24);
            state = (state * 1664525u) + 1013904223u;
            var u2 = (state >> 8) / (float)(1 << 24);
            return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
        }
    }
}
