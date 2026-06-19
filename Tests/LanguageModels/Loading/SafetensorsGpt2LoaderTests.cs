// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Text;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Tests for <see cref="SafetensorsGpt2Loader"/>. The placement checks use a tiny
    /// synthetic GPT-2 (d=4, heads=2, 1 layer, vocab=6) whose tensors are filled with
    /// distinct ramps, so the expected location of every weight after the conv1d
    /// transpose + per-head Q/K/V/O split is computed directly from the input — an
    /// independent reference, not the loader's own logic. A [LongFact] checks real
    /// bit-parity against <c>gpt2_small.bin</c> when a GPT-2 <c>model.safetensors</c>
    /// is present.
    /// </summary>
    public sealed class SafetensorsGpt2LoaderTests
    {
        private const int D = 4, Heads = 2, Layers = 1, Vocab = 6, Ctx = 5, DFF = 16;
        private const int DHead = D / Heads;

        private readonly ITestOutputHelper _output;

        public SafetensorsGpt2LoaderTests(ITestOutputHelper output)
        {
            _output = output;
        }

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

        [Fact]
        public void Load_TinyGpt2_PlacesQkvAndLmHead_Correctly()
        {
            var t = BuildTinyTensors();
            using var reader = new SafetensorsReader(new MemoryStream(BuildSafetensors(t)), ownsStream: true);

            using var model = SafetensorsGpt2Loader.Load(reader, TinyConfig);

            // Re-serialise and parse the parameter stream (order = GPT1Model.Save).
            var prms = SaveAndParse(model);

            var wte = t["wte.weight"];
            var cAttn = t["h.0.attn.c_attn.weight"];   // [D, 3D]
            var cAttnB = t["h.0.attn.c_attn.bias"];     // [3D]
            var cProj = t["h.0.attn.c_proj.weight"];    // [D, D]

            // index map for a 1-layer tiny model
            const int iWte = 0, iWpe = 1, iLn1w = 2;
            int PerHead(int h) => iLn1w + 2 + h * 7;     // first param of head h
            const int iLmHead = 27;

            Assert.Equal(wte, prms[iWte]);
            Assert.Equal(t["wpe.weight"], prms[iWpe]);

            // Head 0 Wq: cols [0, DHead) of the Q block (col offset 0).
            AssertColBlock(cAttn, 3 * D, colOffset: 0 * D + 0 * DHead, prms[PerHead(0) + 0]);
            // Head 1 Wk: K block (offset D), head 1 (col offset D + DHead).
            AssertColBlock(cAttn, 3 * D, colOffset: 1 * D + 1 * DHead, prms[PerHead(1) + 2]);
            // Head 1 Wv bias: V bias block (offset 2D), head 1.
            AssertRange(cAttnB, 2 * D + 1 * DHead, DHead, prms[PerHead(1) + 5]);
            // Head 1 Wo: rows [DHead, 2*DHead) of c_proj [D, D].
            AssertRowBlock(cProj, D, rowOffset: 1 * DHead, prms[PerHead(1) + 6]);

            // LM head = wte transposed to [D, vocab].
            var lm = prms[iLmHead];
            Assert.Equal((long)D * Vocab, lm.Length);
            for (var i = 0; i < D; i++)
            {
                for (var v = 0; v < Vocab; v++)
                {
                    Assert.Equal(wte[v * D + i], lm[i * Vocab + v]);
                }
            }
        }

        [Fact]
        public void Load_TinyGpt2_DecodesFiniteLogits()
        {
            var t = BuildTinyTensors();
            using var reader = new SafetensorsReader(new MemoryStream(BuildSafetensors(t)), ownsStream: true);
            using var model = SafetensorsGpt2Loader.Load(reader, TinyConfig);
            model.Eval();

            using var adapter = new CachedGpt1ModelAdapter(model);
            var logits = new float[Vocab];
            adapter.DecodeNextToken(0, logits);

            for (var i = 0; i < Vocab; i++)
            {
                Assert.False(float.IsNaN(logits[i]) || float.IsInfinity(logits[i]));
            }
        }

        [LongFact]
        public void Load_RealGpt2Safetensors_BitParity_WithBinFixture()
        {
            var safe = ResolveFirst(
                Environment.GetEnvironmentVariable("OVERFIT_MODEL_DIR") is { } d
                    ? Path.Combine(d, "model.safetensors") : null,
                @"C:\gpt2\model.safetensors");
            var bin = ResolveFirst(@"C:\gpt2\gpt2_small.bin");

            if (safe is null || bin is null)
            {
                _output.WriteLine("GPT-2 model.safetensors and/or gpt2_small.bin not found — skipping.");
                return;
            }

            using var model = SafetensorsGpt2Loader.Load(safe, Gpt2Config.Small);

            var tmp = Path.Combine(Path.GetTempPath(), $"overfit_gpt2_from_safetensors_{Guid.NewGuid():N}.bin");
            try
            {
                using (var fs = File.Create(tmp))
                using (var bw = new BinaryWriter(fs))
                {
                    model.Save(bw);
                }
                Assert.True(FilesEqual(tmp, bin),
                    "safetensors-loaded GPT-2 does not byte-match gpt2_small.bin (convert_gpt2.py output).");
            }
            finally
            {
                if (File.Exists(tmp))
                {
                    File.Delete(tmp);
                }
            }
        }

        // ── synthetic GPT-2 tensors filled with distinct ramps ──────────────
        private static Dictionary<string, float[]> BuildTinyTensors()
        {
            var t = new Dictionary<string, float[]>();
            var seed = 0f;
            float[] Ramp(int n, float baseVal)
            {
                var a = new float[n];
                for (var i = 0; i < n; i++)
                {
                    a[i] = baseVal + i * 0.125f;
                }
                return a;
            }
            void Add(string name, int n)
            {
                t[name] = Ramp(n, seed);
                seed += 1000f;
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

        // Serialises the named tensors (F32) to a safetensors blob, offsets in insertion order.
        private static byte[] BuildSafetensors(Dictionary<string, float[]> tensors)
        {
            // Shapes are flat 1-D here (only element counts matter to the loader's checks).
            var sb = new StringBuilder("{");
            long offset = 0;
            var first = true;
            foreach (var kv in tensors)
            {
                if (!first)
                {
                    sb.Append(',');
                }
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

        private static List<float[]> SaveAndParse(GPT1Model model)
        {
            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                model.Save(bw);
            }
            ms.Position = 0;

            var prms = new List<float[]>();
            using var br = new BinaryReader(ms);
            while (ms.Position < ms.Length)
            {
                var count = br.ReadInt32();
                var arr = new float[count];
                for (var i = 0; i < count; i++)
                {
                    arr[i] = br.ReadSingle();
                }
                prms.Add(arr);
            }
            return prms;
        }

        private static void AssertColBlock(float[] src, int srcCols, int colOffset, float[] actual)
        {
            var rows = src.Length / srcCols;
            Assert.Equal(rows * DHead, actual.Length);
            for (var i = 0; i < rows; i++)
            {
                for (var c = 0; c < DHead; c++)
                {
                    Assert.Equal(src[i * srcCols + colOffset + c], actual[i * DHead + c]);
                }
            }
        }

        private static void AssertRowBlock(float[] src, int cols, int rowOffset, float[] actual)
        {
            Assert.Equal(DHead * cols, actual.Length);
            for (var r = 0; r < DHead; r++)
            {
                for (var c = 0; c < cols; c++)
                {
                    Assert.Equal(src[(rowOffset + r) * cols + c], actual[r * cols + c]);
                }
            }
        }

        private static void AssertRange(float[] src, int offset, int len, float[] actual)
        {
            Assert.Equal(len, actual.Length);
            for (var i = 0; i < len; i++)
            {
                Assert.Equal(src[offset + i], actual[i]);
            }
        }

        private static string? ResolveFirst(params string?[] candidates)
        {
            foreach (var c in candidates)
            {
                if (c is not null && File.Exists(c))
                {
                    return c;
                }
            }
            return null;
        }

        private static bool FilesEqual(string a, string b)
        {
            var fa = new FileInfo(a);
            var fb = new FileInfo(b);
            if (fa.Length != fb.Length)
            {
                return false;
            }

            using var sa = fa.OpenRead();
            using var sb = fb.OpenRead();
            var bufA = new byte[1 << 16];
            var bufB = new byte[1 << 16];
            int n;
            while ((n = sa.ReadAtLeast(bufA, bufA.Length, throwOnEndOfStream: false)) > 0)
            {
                sb.ReadExactly(bufB, 0, n);
                if (!bufA.AsSpan(0, n).SequenceEqual(bufB.AsSpan(0, n)))
                {
                    return false;
                }
            }
            return true;
        }
    }
}
