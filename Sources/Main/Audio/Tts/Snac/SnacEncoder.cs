// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts.Snac
{
    /// <summary>
    /// The SNAC codec encoder — a 24 kHz waveform to residual-VQ codes (the missing half of the codec; the decoder
    /// is <see cref="SnacDecoder"/>). Wired to the reference graph: input conv → four downsampling
    /// <c>EncoderBlock</c>s (3 dilated <see cref="SnacBlocks.ResidualUnit"/>s → Snake → strided conv) → depthwise
    /// output conv → latent; then per VQ level: avg-pool → <c>in_proj</c> → nearest-codebook assignment →
    /// residual subtraction. Pure managed; unlocks pure-.NET voice-clone dataset prep (audio → tokens).
    /// </summary>
    internal sealed class SnacEncoder
    {
        private static readonly int[] ResidualDilations = [1, 3, 9];

        private readonly SnacConfig _cfg;
        private readonly SnacWeights _w;

        public SnacEncoder(SnacConfig config, SnacWeights weights)
        {
            _cfg = config;
            _w = weights;
        }

        /// <summary>
        /// Encodes mono 24 kHz PCM into one int-code array per VQ level (coarse→fine). Input is zero-padded up to
        /// the codec's frame alignment, as in the reference.
        /// </summary>
        public int[][] Encode(ReadOnlySpan<float> audio)
        {
            var z = RunEncoder(audio, out var frames);
            return Quantize(z, frames);
        }

        // Encoder.model: in_conv → EncoderBlocks → depthwise out_conv → latent [LatentDim × frames]
        private float[] RunEncoder(ReadOnlySpan<float> audio, out int frames)
        {
            var padded = Preprocess(audio);
            var tIn = padded.Length;

            // enc.in_conv: 1 → EncoderDim, k7 pad3 (length preserved)
            var dim = _cfg.EncoderDim;
            var x = new float[dim * tIn];
            SnacConv.Conv1d(padded, _w["enc.in_conv.weight"], _w["enc.in_conv.bias"], x,
                inC: 1, tIn: tIn, outC: dim, kSize: 7, stride: 1, pad: 3, dilation: 1, groups: 1, tOut: tIn);

            var curT = tIn;
            for (var b = 0; b < _cfg.EncoderRates.Length; b++)
            {
                x = EncoderBlock(x, dim, curT, b, out var outDim, out var outT);
                dim = outDim;
                curT = outT;
            }

            // enc.out_conv: depthwise k7 pad3 (length preserved)
            var outConv = new float[dim * curT];
            SnacConv.Conv1d(x, _w["enc.out_conv.weight"], _w["enc.out_conv.bias"], outConv,
                inC: dim, tIn: curT, outC: dim, kSize: 7, stride: 1, pad: 3, dilation: 1, groups: dim, tOut: curT);

            frames = curT;
            return outConv;
        }

        // EncoderBlock: 3 ResidualUnits (dilations 1/3/9, depthwise) → Snake → strided down-conv (k=2·stride).
        private float[] EncoderBlock(float[] x, int inDim, int tIn, int b, out int outDim, out int outT)
        {
            for (var r = 0; r < ResidualDilations.Length; r++)
            {
                x = SnacBlocks.ResidualUnit(_w, $"enc.block.{b}.res.{r}.", x, inDim, tIn, ResidualDilations[r]);
            }

            SnacActivations.Snake1dInPlace(x, _w[$"enc.block.{b}.snake.alpha"], inDim, tIn);

            outDim = inDim * 2;
            var stride = _cfg.EncoderRates[b];
            var k = 2 * stride;
            var pad = (stride + 1) / 2; // ceil(stride/2)
            outT = SnacConv.ConvOutputLength(tIn, k, stride, pad, dilation: 1);

            var down = new float[outDim * outT];
            SnacConv.Conv1d(x, _w[$"enc.block.{b}.down.weight"], _w[$"enc.block.{b}.down.bias"], down,
                inC: inDim, tIn: tIn, outC: outDim, kSize: k, stride: stride, pad: pad, dilation: 1, groups: 1, tOut: outT);
            return down;
        }

        // ResidualVectorQuantize.forward (encode): per level avg-pool → in_proj → nearest code → subtract residual.
        private int[][] Quantize(float[] z, int frames)
        {
            var latent = _cfg.LatentDim;
            var cbDim = _cfg.CodebookDim;

            var residual = new float[z.Length];
            z.AsSpan().CopyTo(residual);

            var codes = new int[_cfg.CodebookCount][];
            for (var i = 0; i < codes.Length; i++)
            {
                var stride = _cfg.VqStrides[i];
                var ti = frames / stride;

                // avg_pool1d(stride) when the level is coarser than the latent rate
                float[] pooled;
                if (stride > 1)
                {
                    pooled = new float[latent * ti];
                    SnacResidualVq.AveragePoolTime(residual, pooled, latent, frames, stride);
                }
                else
                {
                    pooled = residual;
                }

                // in_proj: latent → codebook_dim (1×1)
                var zE = new float[cbDim * ti];
                SnacConv.Conv1d(pooled, _w[$"q.{i}.in_proj.weight"], _w[$"q.{i}.in_proj.bias"], zE,
                    inC: latent, tIn: ti, outC: cbDim, kSize: 1, stride: 1, pad: 0, dilation: 1, groups: 1, tOut: ti);

                // nearest codebook entry per frame → the codes
                var indices = new int[ti];
                var codebook = _w[$"q.{i}.codebook"];
                SnacResidualVq.EncodeCodebook(zE, cbDim, ti, codebook, _cfg.CodebookSize, indices);
                codes[i] = indices;

                // reconstruct this level's contribution and remove it from the residual for the next level
                var zqLow = new float[cbDim * ti];
                SnacResidualVq.DecodeCodebook(indices, codebook, zqLow, _cfg.CodebookSize, cbDim, ti);

                var zq = new float[latent * ti];
                SnacConv.Conv1d(zqLow, _w[$"q.{i}.out_proj.weight"], _w[$"q.{i}.out_proj.bias"], zq,
                    inC: cbDim, tIn: ti, outC: latent, kSize: 1, stride: 1, pad: 0, dilation: 1, groups: 1, tOut: ti);

                if (stride > 1)
                {
                    var up = new float[latent * frames];
                    SnacResidualVq.RepeatInterleaveTime(zq, up, latent, ti, stride);
                    for (var j = 0; j < residual.Length; j++)
                    {
                        residual[j] -= up[j];
                    }
                }
                else
                {
                    for (var j = 0; j < residual.Length; j++)
                    {
                        residual[j] -= zq[j];
                    }
                }
            }

            return codes;
        }

        // SNAC preprocess: right-pad with zeros to a multiple of hop_length · lcm(vq_strides[0], 1).
        private float[] Preprocess(ReadOnlySpan<float> audio)
        {
            var padTo = _cfg.HopLength * _cfg.VqStrides[0]; // attn_window_size is null → lcm(stride0, 1) = stride0
            var padded = ((audio.Length + padTo - 1) / padTo) * padTo;
            var buffer = new float[padded];
            audio.CopyTo(buffer);
            return buffer;
        }
    }
}
