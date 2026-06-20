// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Embeddings;
using DevOnBike.Overfit.Tests.TestSupport.Helpers;

namespace DevOnBike.Overfit.Tests.LanguageModels.Embeddings
{
    /// <summary>
    /// Offline validation of <see cref="BertSafetensorsLoader"/> — fabricates a HuggingFace-layout
    /// safetensors file whose every element encodes its (out, in) source index, loads it, and asserts
    /// that the transpose ([out,in] → [in,out]) and the per-head Q/K/V/O split placed each element in
    /// the correct Overfit parameter slot. This checks the mapping formula directly (not via a circular
    /// forward-equality), so a transpose bug fails loudly. Real-weight parity is the MiniLM [LongFact].
    /// </summary>
    public sealed class BertSafetensorsLoaderTests
    {
        private const int D = 8, Layers = 2, Heads = 2, DFF = 16, MaxPos = 16, Vocab = 20;
        private const int DHead = D / Heads;

        private static BertConfig Config() => new(D, Layers, Heads, DFF, MaxPos, Vocab, typeVocabSize: 2, layerNormEps: 1e-12f);

        // Weight element encoding: a 2-D [rows, cols] tensor stores rows*1000 + cols at [r, c].
        private static float[] Encode2D(int rows, int cols)
        {
            var a = new float[rows * cols];
            for (var r = 0; r < rows; r++)
            {
                for (var c = 0; c < cols; c++)
                {
                    a[r * cols + c] = r * 1000 + c;
                }
            }

            return a;
        }

        private static float[] Iota(int n)
        {
            var a = new float[n];
            for (var i = 0; i < n; i++)
            {
                a[i] = i;
            }
            return a;
        }

        private static float[] Const(int n, float v)
        {
            var a = new float[n];
            Array.Fill(a, v);
            return a;
        }

        private static string WriteFixture(string prefix)
        {
            var w = new SafetensorsTestWriter();
            w.Add(prefix + "embeddings.word_embeddings.weight", new long[] { Vocab, D }, Encode2D(Vocab, D));
            w.Add(prefix + "embeddings.position_embeddings.weight", new long[] { MaxPos, D }, Encode2D(MaxPos, D));
            w.Add(prefix + "embeddings.token_type_embeddings.weight", new long[] { 2, D }, Encode2D(2, D));
            w.Add(prefix + "embeddings.LayerNorm.weight", new long[] { D }, Const(D, 1f));
            w.Add(prefix + "embeddings.LayerNorm.bias", new long[] { D }, Const(D, 0f));

            for (var i = 0; i < Layers; i++)
            {
                var lp = prefix + "encoder.layer." + i + ".";
                w.Add(lp + "attention.self.query.weight", new long[] { D, D }, Encode2D(D, D));
                w.Add(lp + "attention.self.query.bias", new long[] { D }, Iota(D));
                w.Add(lp + "attention.self.key.weight", new long[] { D, D }, Encode2D(D, D));
                w.Add(lp + "attention.self.key.bias", new long[] { D }, Iota(D));
                w.Add(lp + "attention.self.value.weight", new long[] { D, D }, Encode2D(D, D));
                w.Add(lp + "attention.self.value.bias", new long[] { D }, Iota(D));
                w.Add(lp + "attention.output.dense.weight", new long[] { D, D }, Encode2D(D, D));
                w.Add(lp + "attention.output.dense.bias", new long[] { D }, Iota(D));
                w.Add(lp + "attention.output.LayerNorm.weight", new long[] { D }, Const(D, 1f));
                w.Add(lp + "attention.output.LayerNorm.bias", new long[] { D }, Const(D, 0f));
                w.Add(lp + "intermediate.dense.weight", new long[] { DFF, D }, Encode2D(DFF, D));
                w.Add(lp + "intermediate.dense.bias", new long[] { DFF }, Iota(DFF));
                w.Add(lp + "output.dense.weight", new long[] { D, DFF }, Encode2D(D, DFF));
                w.Add(lp + "output.dense.bias", new long[] { D }, Iota(D));
                w.Add(lp + "output.LayerNorm.weight", new long[] { D }, Const(D, 1f));
                w.Add(lp + "output.LayerNorm.bias", new long[] { D }, Const(D, 0f));
            }

            var path = Path.Combine(Path.GetTempPath(), "overfit_bert_" + Guid.NewGuid().ToString("N") + ".safetensors");
            w.Write(path);
            return path;
        }

        [Fact]
        public void Load_PlacesEmbeddingsAndTransposesAttentionAndFfnCorrectly()
        {
            var path = WriteFixture(prefix: "");
            try
            {
                using var enc = BertSafetensorsLoader.Load(path, Config());

                // Embeddings copy straight through: [v, j] = v*1000 + j.
                var word = enc.WordEmbeddings.Weight.DataReadOnlySpan;
                Assert.Equal(5 * 1000 + 3, word[5 * D + 3]);

                var block = enc.Layers[0];
                var attn = block.Attention;

                // Q per-head: WqHeads[h][i, j] comes from HF query.weight[out=h*dHead+j, in=i] = (h*dHead+j)*1000 + i.
                for (var h = 0; h < Heads; h++)
                {
                    var wq = attn.WqHeads[h].DataReadOnlySpan; // [D, dHead]
                    for (var i = 0; i < D; i++)
                    {
                        for (var j = 0; j < DHead; j++)
                        {
                            Assert.Equal((h * DHead + j) * 1000 + i, wq[i * DHead + j]);
                        }
                    }

                    // Q bias per head: BqHeads[h][j] = HF query.bias[h*dHead + j].
                    var bq = attn.BqHeads[h].DataReadOnlySpan;
                    for (var j = 0; j < DHead; j++)
                    {
                        Assert.Equal(h * DHead + j, bq[j]);
                    }
                }

                // Output projection: WoHeads[h][r, c] from HF dense.weight[out=c, in=h*dHead+r] = c*1000 + (h*dHead+r).
                for (var h = 0; h < Heads; h++)
                {
                    var wo = attn.WoHeads[h].DataReadOnlySpan; // [dHead, D]
                    for (var r = 0; r < DHead; r++)
                    {
                        for (var c = 0; c < D; c++)
                        {
                            Assert.Equal(c * 1000 + (h * DHead + r), wo[r * D + c]);
                        }
                    }
                }

                // Output bias → single Bo [D].
                var bo = attn.Bo.DataReadOnlySpan;
                for (var c = 0; c < D; c++)
                {
                    Assert.Equal(c, bo[c]);
                }

                // FFN W1 [d, dFF]: W1[i, j] from HF intermediate.weight[out=j, in=i] = j*1000 + i.
                var w1 = block.FFN.W1.DataReadOnlySpan;
                for (var i = 0; i < D; i++)
                {
                    for (var j = 0; j < DFF; j++)
                    {
                        Assert.Equal(j * 1000 + i, w1[i * DFF + j]);
                    }
                }

                // FFN W2 [dFF, d]: W2[i, j] from HF output.weight[out=j, in=i] = j*1000 + i.
                var w2 = block.FFN.W2.DataReadOnlySpan;
                for (var i = 0; i < DFF; i++)
                {
                    for (var j = 0; j < D; j++)
                    {
                        Assert.Equal(j * 1000 + i, w2[i * D + j]);
                    }
                }
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void Load_DetectsBertPrefix()
        {
            var path = WriteFixture(prefix: "bert.");
            try
            {
                using var enc = BertSafetensorsLoader.Load(path, Config());
                var word = enc.WordEmbeddings.Weight.DataReadOnlySpan;
                Assert.Equal(7 * 1000 + 2, word[7 * D + 2]);
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void Load_ThenEmbed_ProducesUnitNormVector()
        {
            var path = WriteFixture(prefix: "");
            try
            {
                using var enc = BertSafetensorsLoader.Load(path, Config());
                var v = enc.Embed(new[] { 2, 5, 9, 3 });
                var norm = 0f;
                foreach (var x in v)
                {
                    norm += x * x;
                }
                Assert.True(MathF.Abs(MathF.Sqrt(norm) - 1f) < 1e-4f);
            }
            finally
            {
                File.Delete(path);
            }
        }
    }
}
