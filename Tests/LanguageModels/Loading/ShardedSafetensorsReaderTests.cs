// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Text;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Tests for sharded HuggingFace safetensors (<c>model.safetensors.index.json</c> +
    /// <c>model-0000k-of-0000n.safetensors</c>): the <see cref="ShardedSafetensorsReader"/>
    /// merges shards transparently, and a sharded GPT-2 loads byte-identically to the
    /// same model in a single file.
    /// </summary>
    public sealed class ShardedSafetensorsReaderTests
    {
        [Fact]
        public void Sharded_MergesTensors_AndReadsFromCorrectShard()
        {
            var shard1 = new Dictionary<string, float[]> { ["a.weight"] = [1f, 2f, 3f] };
            var shard2 = new Dictionary<string, float[]> { ["b.weight"] = [4f, 5f], ["c.bias"] = [6f] };

            var dir = WriteShardedRepo(("model-00001-of-00002.safetensors", shard1),
                                       ("model-00002-of-00002.safetensors", shard2));
            try
            {
                using var src = SafetensorsSource.Open(dir);
                var sharded = Assert.IsType<ShardedSafetensorsReader>(src);

                Assert.Equal(2, sharded.ShardCount);
                Assert.Equal(3, src.Tensors.Count);
                Assert.Equal(3, src.ElementCount("a.weight"));
                Assert.Equal(2, src.ElementCount("b.weight"));

                var a = new float[3];
                src.LoadF32("a.weight", a);
                Assert.Equal(new[] { 1f, 2f, 3f }, a);

                var b = new float[2];
                src.LoadF32("b.weight", b);
                Assert.Equal(new[] { 4f, 5f }, b);

                var c = new float[1];
                src.LoadF32("c.bias", c);
                Assert.Equal(new[] { 6f }, c);
            }
            finally
            {
                Directory.Delete(dir, recursive: true);
            }
        }

        [Fact]
        public void Sharded_WeightMapPointsToMissingTensor_Throws()
        {
            var dir = Directory.CreateTempSubdirectory("overfit_st_bad_").FullName;
            try
            {
                File.WriteAllBytes(Path.Combine(dir, "model-00001-of-00001.safetensors"),
                    BuildSafetensors(new Dictionary<string, float[]> { ["present"] = [1f] }));
                File.WriteAllText(Path.Combine(dir, "model.safetensors.index.json"),
                    "{\"weight_map\":{\"absent\":\"model-00001-of-00001.safetensors\"}}");

                Assert.Throws<OverfitFormatException>(() => SafetensorsSource.Open(dir));
            }
            finally
            {
                Directory.Delete(dir, recursive: true);
            }
        }

        [Fact]
        public void Sharded_Gpt2_LoadsByteIdentical_ToSingleFile()
        {
            var tensors = BuildTinyGpt2Tensors();
            var cfg = TinyConfig;

            // Single file.
            byte[] single;
            using (var src = new SafetensorsReader(new MemoryStream(BuildSafetensors(tensors)), ownsStream: true))
            using (var model = SafetensorsGpt2Loader.Load(src, cfg))
            {
                single = SaveBytes(model);
            }

            // Same tensors split across two shards.
            var half = tensors.Count / 2;
            var s1 = new Dictionary<string, float[]>();
            var s2 = new Dictionary<string, float[]>();
            var i = 0;
            foreach (var kv in tensors)
            {
                (i++ < half ? s1 : s2)[kv.Key] = kv.Value;
            }

            var dir = WriteShardedRepo(("model-00001-of-00002.safetensors", s1),
                                       ("model-00002-of-00002.safetensors", s2));
            try
            {
                using var src = SafetensorsSource.Open(dir);
                using var model = SafetensorsGpt2Loader.Load(src, cfg);
                var sharded = SaveBytes(model);
                Assert.Equal(single, sharded);
            }
            finally
            {
                Directory.Delete(dir, recursive: true);
            }
        }

        // ── helpers ─────────────────────────────────────────────────────────
        private const int D = 4, Heads = 2, Layers = 1, Vocab = 6, Ctx = 5, DFF = 16;

        private static GPT1Config TinyConfig => new()
        {
            VocabSize = Vocab,
            ContextLength = Ctx,
            DModel = D,
            NHeads = Heads,
            NLayers = Layers,
            DFF = DFF,
            TieWeights = false,
            PreLayerNorm = true,
        };

        private static Dictionary<string, float[]> BuildTinyGpt2Tensors()
        {
            var t = new Dictionary<string, float[]>();
            var seed = 0f;
            void Add(string name, int n)
            {
                var a = new float[n];
                for (var i = 0; i < n; i++) { a[i] = seed + i * 0.125f; }
                seed += 1000f;
                t[name] = a;
            }

            Add("wte.weight", Vocab * D);
            Add("wpe.weight", Ctx * D);
            Add("h.0.ln_1.weight", D);
            Add("h.0.ln_1.bias", D);
            Add("h.0.attn.c_attn.weight", D * 3 * D);
            Add("h.0.attn.c_attn.bias", 3 * D);
            Add("h.0.attn.c_proj.weight", D * D);
            Add("h.0.attn.c_proj.bias", D);
            Add("h.0.ln_2.weight", D);
            Add("h.0.ln_2.bias", D);
            Add("h.0.mlp.c_fc.weight", D * DFF);
            Add("h.0.mlp.c_fc.bias", DFF);
            Add("h.0.mlp.c_proj.weight", DFF * D);
            Add("h.0.mlp.c_proj.bias", D);
            Add("ln_f.weight", D);
            Add("ln_f.bias", D);
            return t;
        }

        private static byte[] SaveBytes(GPT1Model model)
        {
            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                model.Save(bw);
            }
            return ms.ToArray();
        }

        // Writes shard files + a model.safetensors.index.json into a fresh temp dir; returns the dir.
        private static string WriteShardedRepo(params (string file, Dictionary<string, float[]> tensors)[] shards)
        {
            var dir = Directory.CreateTempSubdirectory("overfit_st_shard_").FullName;

            var weightMap = new StringBuilder("{\"weight_map\":{");
            var first = true;
            foreach (var (file, tensors) in shards)
            {
                File.WriteAllBytes(Path.Combine(dir, file), BuildSafetensors(tensors));
                foreach (var name in tensors.Keys)
                {
                    if (!first) { weightMap.Append(','); }
                    first = false;
                    weightMap.Append('"').Append(name).Append("\":\"").Append(file).Append('"');
                }
            }
            weightMap.Append("}}");
            File.WriteAllText(Path.Combine(dir, "model.safetensors.index.json"), weightMap.ToString());

            return dir;
        }

        private static byte[] BuildSafetensors(Dictionary<string, float[]> tensors)
        {
            var sb = new StringBuilder("{");
            long offset = 0;
            var first = true;
            foreach (var kv in tensors)
            {
                if (!first) { sb.Append(','); }
                first = false;
                var bytes = kv.Value.Length * 4L;
                sb.Append('"').Append(kv.Key).Append("\":{\"dtype\":\"F32\",\"shape\":[")
                  .Append(kv.Value.Length).Append("],\"data_offsets\":[")
                  .Append(offset).Append(',').Append(offset + bytes).Append("]}");
                offset += bytes;
            }
            sb.Append('}');

            var header = Encoding.UTF8.GetBytes(sb.ToString());
            using var ms = new MemoryStream();
            Span<byte> len = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(len, (ulong)header.Length);
            ms.Write(len);
            ms.Write(header);
            Span<byte> tmp = stackalloc byte[4];
            foreach (var kv in tensors)
            {
                foreach (var f in kv.Value)
                {
                    BinaryPrimitives.WriteUInt32LittleEndian(tmp, BitConverter.SingleToUInt32Bits(f));
                    ms.Write(tmp);
                }
            }
            return ms.ToArray();
        }
    }
}
