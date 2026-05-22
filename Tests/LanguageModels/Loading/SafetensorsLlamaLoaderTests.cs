// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Text;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Validates <see cref="SafetensorsLlamaLoader"/> without a real multi-GB model:
    /// the same random weights are emitted twice — once as an Overfit <c>.bin</c>
    /// (the validated <see cref="CachedLlamaInferenceEngine.Load"/> reference) and
    /// once as a HuggingFace <c>safetensors</c> blob in the transposed HF layout.
    /// Both are loaded, prefilled with the same tokens, and their end-of-prompt
    /// logits must be bit-identical — which only holds if the loader's per-head /
    /// FFN / LM-head transposes invert the HF layout exactly.
    ///
    /// GQA is exercised (nKvHeads &lt; nHeads); headDim (4) is deliberately not a
    /// Q8 block multiple so the F32 path is what's compared (bit-exact).
    /// </summary>
    public sealed class SafetensorsLlamaLoaderTests
    {
        private const int Vocab = 12;
        private const int D = 8;
        private const int NHeads = 2;
        private const int NKv = 1;          // GQA
        private const int Hd = D / NHeads;  // 4
        private const int DFF = 16;
        private const int Layers = 2;
        private const int Ctx = 16;
        private const float RopeTheta = 10_000f;

        private static GPT1Config Config() => new()
        {
            NLayers = Layers,
            DModel = D,
            NHeads = NHeads,
            NKvHeads = NKv,
            VocabSize = Vocab,
            ContextLength = Ctx,
            DFF = DFF,
            UseRoPE = true,
            RoPETheta = RopeTheta,
            FfnActivation = FeedForwardActivation.SwiGLU,
            TieWeights = false,
        };

        [Fact]
        public void Load_Safetensors_MatchesBinReference_BitIdentical()
        {
            var w = new Weights(seed: 1234);

            using var binEngine = LoadFromBin(w);
            using var stEngine = LoadFromSafetensors(w, quantize: false);

            int[] tokens = [3, 7, 0, 11, 5, 2];
            using var binSession = binEngine.CreateSession(Ctx);
            using var stSession = stEngine.CreateSession(Ctx);
            binSession.Reset(tokens);
            stSession.Reset(tokens);

            var expected = binSession.LastLogits;
            var actual = stSession.LastLogits;

            Assert.Equal(Vocab, expected.Length);
            Assert.Equal(Vocab, actual.Length);
            for (var i = 0; i < Vocab; i++)
            {
                Assert.Equal(expected[i], actual[i]);   // bit-identical F32
            }
        }

        [Fact]
        public void Load_Safetensors_Quantized_IsCloseToReference()
        {
            // headDim=4 is below the Q8 block size, so attention stays F32 even with
            // quantize:true; FFN/LM-head (dims 8/16/12) are also sub-block, so the
            // whole model loads F32 — quantize:true must be a no-op here, bit-exact.
            var w = new Weights(seed: 99);

            using var binEngine = LoadFromBin(w);
            using var stEngine = LoadFromSafetensors(w, quantize: true);

            int[] tokens = [1, 4, 9, 2];
            using var binSession = binEngine.CreateSession(Ctx);
            using var stSession = stEngine.CreateSession(Ctx);
            binSession.Reset(tokens);
            stSession.Reset(tokens);

            for (var i = 0; i < Vocab; i++)
            {
                Assert.Equal(binSession.LastLogits[i], stSession.LastLogits[i]);
            }
        }

        // ── Reference .bin path (CachedLlamaInferenceEngine.Load) ───────────────
        private static CachedLlamaInferenceEngine LoadFromBin(Weights w)
        {
            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                bw.Write(0x4F565246u);                          // magic "OVRF"
                bw.Write(2);                                    // version
                bw.Write(Layers);
                bw.Write(D);
                bw.Write(NHeads);
                bw.Write(NKv);
                bw.Write(Vocab);
                bw.Write(Ctx);
                bw.Write(DFF);
                bw.Write(1);                                    // use_rope
                bw.Write(RopeTheta);
                bw.Write((int)FeedForwardActivation.SwiGLU);
                bw.Write(0);                                    // tie_weights = false

                WriteRaw(bw, w.Embed);
                for (var l = 0; l < Layers; l++)
                {
                    WriteRaw(bw, w.AttnNorm[l]);
                    WriteRaw(bw, new float[D]);                 // attn norm beta
                    // Q/K are RoPE-rotated: the runtime (and thus the .bin's kernel
                    // weight) is the adjacent-pair layout, which is the headDim permute
                    // of the HF rotate-half weights the safetensors blob stores. V is
                    // not rotated. (This mirrors SafetensorsLlamaLoader's ropePermute.)
                    for (var h = 0; h < NHeads; h++)
                    {
                        WriteRaw(bw, PermuteHeadDim(w.Wq[l][h]));
                        WriteRaw(bw, new float[Hd]);            // bq
                    }
                    for (var kv = 0; kv < NKv; kv++)
                    {
                        WriteRaw(bw, PermuteHeadDim(w.Wk[l][kv]));
                        WriteRaw(bw, new float[Hd]);            // bk
                        WriteRaw(bw, w.Wv[l][kv]);
                        WriteRaw(bw, new float[Hd]);            // bv
                    }
                    for (var h = 0; h < NHeads; h++)
                    {
                        WriteRaw(bw, w.Wo[l][h]);
                        WriteRaw(bw, new float[D]);             // bo
                    }
                    WriteRaw(bw, w.FfnNorm[l]);
                    WriteRaw(bw, new float[D]);                 // ffn norm beta
                    WriteRaw(bw, w.FfnGate[l]);
                    WriteRaw(bw, w.FfnUp[l]);
                    WriteRaw(bw, w.FfnDown[l]);
                }
                WriteRaw(bw, w.FinalNorm);
                WriteRaw(bw, new float[D]);                     // final norm beta
                WriteRaw(bw, w.LmHead);
            }

            ms.Position = 0;
            using var br = new BinaryReader(ms);
            return CachedLlamaInferenceEngine.Load(br);
        }

        // ── HF safetensors path (transposed layout) ─────────────────────────────
        private static CachedLlamaInferenceEngine LoadFromSafetensors(Weights w, bool quantize)
        {
            var entries = new List<(string Name, long[] Shape, float[] Data)>
            {
                ("model.embed_tokens.weight", [Vocab, D], w.Embed),
            };

            for (var l = 0; l < Layers; l++)
            {
                var p = $"model.layers.{l}";
                entries.Add(($"{p}.input_layernorm.weight", [D], w.AttnNorm[l]));

                // q/k/v_proj: HF [heads*headDim, dModel], hf[h*hd+j, i] = Wkernel[i*hd+j].
                entries.Add(($"{p}.self_attn.q_proj.weight", [NHeads * Hd, D], HfQkv(w.Wq[l], NHeads)));
                entries.Add(($"{p}.self_attn.k_proj.weight", [NKv * Hd, D], HfQkv(w.Wk[l], NKv)));
                entries.Add(($"{p}.self_attn.v_proj.weight", [NKv * Hd, D], HfQkv(w.Wv[l], NKv)));

                // o_proj: HF [dModel, heads*headDim], hf[j, h*hd+i] = Wo_kernel[i*d+j].
                entries.Add(($"{p}.self_attn.o_proj.weight", [D, NHeads * Hd], HfOutput(w.Wo[l])));

                entries.Add(($"{p}.post_attention_layernorm.weight", [D], w.FfnNorm[l]));

                // gate/up_proj: HF [dFF, dModel] = kernel[dModel, dFF].T
                entries.Add(($"{p}.mlp.gate_proj.weight", [DFF, D], HfTranspose(w.FfnGate[l], D, DFF)));
                entries.Add(($"{p}.mlp.up_proj.weight", [DFF, D], HfTranspose(w.FfnUp[l], D, DFF)));
                // down_proj: HF [dModel, dFF] = kernel[dFF, dModel].T
                entries.Add(($"{p}.mlp.down_proj.weight", [D, DFF], HfTranspose(w.FfnDown[l], DFF, D)));
            }

            entries.Add(("model.norm.weight", [D], w.FinalNorm));
            entries.Add(("lm_head.weight", [Vocab, D], w.LmHead));

            using var source = new SafetensorsReader(BuildSafetensors(entries), ownsStream: true);
            return SafetensorsLlamaLoader.Load(source, Config(), quantize);
        }

        // kernel per-head [d, hd] (flat i*hd+j) → HF [count*hd, d] (flat (h*hd+j)*d + i)
        private static float[] HfQkv(float[][] heads, int count)
        {
            var outBuf = new float[count * Hd * D];
            for (var h = 0; h < count; h++)
            {
                for (var i = 0; i < D; i++)
                {
                    for (var j = 0; j < Hd; j++)
                    {
                        outBuf[(h * Hd + j) * D + i] = heads[h][i * Hd + j];
                    }
                }
            }
            return outBuf;
        }

        // kernel per-head [hd, d] (flat i*d+j) → HF [d, heads*hd] (flat j*(heads*hd) + h*hd + i)
        private static float[] HfOutput(float[][] heads)
        {
            var nHeadsHd = NHeads * Hd;
            var outBuf = new float[D * nHeadsHd];
            for (var h = 0; h < NHeads; h++)
            {
                for (var i = 0; i < Hd; i++)
                {
                    for (var j = 0; j < D; j++)
                    {
                        outBuf[j * nHeadsHd + h * Hd + i] = heads[h][i * D + j];
                    }
                }
            }
            return outBuf;
        }

        // kernel [inDim, outDim] (flat i*outDim+o) → HF [outDim, inDim] (flat o*inDim+i)
        private static float[] HfTranspose(float[] kernel, int inDim, int outDim)
        {
            var outBuf = new float[inDim * outDim];
            for (var i = 0; i < inDim; i++)
            {
                for (var o = 0; o < outDim; o++)
                {
                    outBuf[o * inDim + i] = kernel[i * outDim + o];
                }
            }
            return outBuf;
        }

        // kernel [D, Hd] → permuted on the headDim axis (out col 2k ← col k, 2k+1 ← col k+Hd/2):
        // the adjacent-pair layout RopeKernel expects from HF rotate-half weights.
        private static float[] PermuteHeadDim(float[] kernel)
        {
            var half = Hd / 2;
            var outBuf = new float[D * Hd];
            for (var i = 0; i < D; i++)
            {
                for (var k = 0; k < half; k++)
                {
                    outBuf[i * Hd + 2 * k] = kernel[i * Hd + k];
                    outBuf[i * Hd + 2 * k + 1] = kernel[i * Hd + k + half];
                }
            }
            return outBuf;
        }

        private static void WriteRaw(BinaryWriter bw, float[] data)
        {
            foreach (var f in data)
            {
                bw.Write(f);
            }
        }

        // Builds an in-memory safetensors blob: 8-byte LE header length + JSON header + F32 data.
        private static MemoryStream BuildSafetensors(List<(string Name, long[] Shape, float[] Data)> entries)
        {
            var sb = new StringBuilder("{");
            long offset = 0;
            for (var e = 0; e < entries.Count; e++)
            {
                var (name, shape, data) = entries[e];
                var begin = offset;
                var end = offset + (long)data.Length * 4;
                offset = end;

                if (e > 0) { sb.Append(','); }
                sb.Append('"').Append(name).Append("\":{\"dtype\":\"F32\",\"shape\":[");
                for (var s = 0; s < shape.Length; s++)
                {
                    if (s > 0) { sb.Append(','); }
                    sb.Append(shape[s]);
                }
                sb.Append("],\"data_offsets\":[").Append(begin).Append(',').Append(end).Append("]}");
            }
            sb.Append('}');

            var headerBytes = Encoding.UTF8.GetBytes(sb.ToString());
            using var ms = new MemoryStream();
            Span<byte> len = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(len, (ulong)headerBytes.Length);
            ms.Write(len);
            ms.Write(headerBytes);

            Span<byte> tmp = stackalloc byte[4];
            foreach (var (_, _, data) in entries)
            {
                foreach (var f in data)
                {
                    BinaryPrimitives.WriteUInt32LittleEndian(tmp, BitConverter.SingleToUInt32Bits(f));
                    ms.Write(tmp);
                }
            }
            return new MemoryStream(ms.ToArray());
        }

        // ── Random weights in kernel layout (the .bin / runtime layout) ─────────
        private sealed class Weights
        {
            public readonly float[] Embed;
            public readonly float[][] AttnNorm;     // [layer][d]
            public readonly float[][][] Wq;         // [layer][head][d*hd]
            public readonly float[][][] Wk;         // [layer][kv][d*hd]
            public readonly float[][][] Wv;
            public readonly float[][][] Wo;         // [layer][head][hd*d]
            public readonly float[][] FfnNorm;
            public readonly float[][] FfnGate;      // [layer][d*dff]
            public readonly float[][] FfnUp;
            public readonly float[][] FfnDown;      // [layer][dff*d]
            public readonly float[] FinalNorm;
            public readonly float[] LmHead;         // [vocab*d]

            public Weights(int seed)
            {
                var rng = new Random(seed);
                Embed = Rand(rng, Vocab * D);
                AttnNorm = new float[Layers][];
                Wq = new float[Layers][][];
                Wk = new float[Layers][][];
                Wv = new float[Layers][][];
                Wo = new float[Layers][][];
                FfnNorm = new float[Layers][];
                FfnGate = new float[Layers][];
                FfnUp = new float[Layers][];
                FfnDown = new float[Layers][];

                for (var l = 0; l < Layers; l++)
                {
                    AttnNorm[l] = Norm(rng);
                    Wq[l] = Heads(rng, NHeads, D * Hd);
                    Wk[l] = Heads(rng, NKv, D * Hd);
                    Wv[l] = Heads(rng, NKv, D * Hd);
                    Wo[l] = Heads(rng, NHeads, Hd * D);
                    FfnNorm[l] = Norm(rng);
                    FfnGate[l] = Rand(rng, D * DFF);
                    FfnUp[l] = Rand(rng, D * DFF);
                    FfnDown[l] = Rand(rng, DFF * D);
                }

                FinalNorm = Norm(rng);
                LmHead = Rand(rng, Vocab * D);
            }

            private static float[][] Heads(Random rng, int count, int size)
            {
                var heads = new float[count][];
                for (var h = 0; h < count; h++) { heads[h] = Rand(rng, size); }
                return heads;
            }

            // RMSNorm gamma near 1 keeps activations sane.
            private static float[] Norm(Random rng)
            {
                var arr = new float[D];
                for (var i = 0; i < D; i++) { arr[i] = 1f + (float)(rng.NextDouble() - 0.5) * 0.1f; }
                return arr;
            }

            private static float[] Rand(Random rng, int n)
            {
                var arr = new float[n];
                for (var i = 0; i < n; i++) { arr[i] = (float)(rng.NextDouble() - 0.5) * 0.2f; }
                return arr;
            }
        }
    }
}
