// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Parses a representative HuggingFace <c>config.json</c> (Qwen2.5-0.5B shape,
    /// with extra nested objects to confirm they're skipped) into a
    /// <see cref="DevOnBike.Overfit.DeepLearning.GPT1Config"/>.
    /// </summary>
    public sealed class LlamaConfigReaderTests
    {
        // Qwen2.5-0.5B-ish, plus nested objects/arrays the reader must skip.
        private const string Qwen05B =
            "{\n" +
            "  \"architectures\": [\"Qwen2ForCausalLM\"],\n" +
            "  \"hidden_size\": 896,\n" +
            "  \"intermediate_size\": 4864,\n" +
            "  \"max_position_embeddings\": 32768,\n" +
            "  \"num_attention_heads\": 14,\n" +
            "  \"num_hidden_layers\": 24,\n" +
            "  \"num_key_value_heads\": 2,\n" +
            "  \"rope_theta\": 1000000.0,\n" +
            "  \"rope_scaling\": {\"type\": \"linear\", \"factor\": 2.0},\n" +
            "  \"tie_word_embeddings\": true,\n" +
            "  \"vocab_size\": 151936,\n" +
            "  \"torch_dtype\": \"bfloat16\"\n" +
            "}";

        [Fact]
        public void Parse_Qwen05B_MapsFieldsAndCapsContext()
        {
            var cfg = LlamaConfigReader.Parse(Encoding.UTF8.GetBytes(Qwen05B));

            Assert.Equal(24, cfg.NLayers);
            Assert.Equal(896, cfg.DModel);
            Assert.Equal(14, cfg.NHeads);
            Assert.Equal(2, cfg.KvHeads);            // GQA
            Assert.Equal(4864, cfg.DFF);
            Assert.Equal(151936, cfg.VocabSize);
            Assert.Equal(8192, cfg.ContextLength);   // capped from 32768
            Assert.Equal(1_000_000f, cfg.RoPETheta);
            Assert.True(cfg.UseRoPE);
            Assert.True(cfg.TieWeights);
            Assert.Equal(FeedForwardActivation.SwiGLU, cfg.FfnActivation);
        }

        [Fact]
        public void Parse_DefaultsKvHeadsToNHeads_WhenAbsent()
        {
            const string json =
                "{\"hidden_size\":64,\"num_attention_heads\":4,\"num_hidden_layers\":2,\"vocab_size\":100}";
            var cfg = LlamaConfigReader.Parse(Encoding.UTF8.GetBytes(json));

            Assert.Equal(4, cfg.KvHeads);            // == NHeads
            Assert.Equal(256, cfg.DFF);              // 4 * hidden_size
            Assert.Equal(10_000f, cfg.RoPETheta);    // default theta
        }

        [Fact]
        public void Parse_RejectsHeadDimMismatch()
        {
            // head_dim * heads (5*4=20) != hidden_size (64) — not representable.
            const string json =
                "{\"hidden_size\":64,\"num_attention_heads\":4,\"num_hidden_layers\":2,\"vocab_size\":100,\"head_dim\":5}";
            Assert.Throws<NotSupportedException>(() => LlamaConfigReader.Parse(Encoding.UTF8.GetBytes(json)));
        }

        [Fact]
        public void Parse_MissingRequiredField_Throws()
        {
            const string json = "{\"hidden_size\":64,\"num_attention_heads\":4}";  // no layers/vocab
            Assert.Throws<InvalidDataException>(() => LlamaConfigReader.Parse(Encoding.UTF8.GetBytes(json)));
        }

        [Fact]
        public void Parse_Llama3RopeScaling_Populated()
        {
            const string json =
                "{\"hidden_size\":64,\"num_attention_heads\":4,\"num_hidden_layers\":2,\"vocab_size\":100," +
                "\"rope_scaling\":{\"factor\":32.0,\"high_freq_factor\":4.0,\"low_freq_factor\":1.0," +
                "\"original_max_position_embeddings\":8192,\"rope_type\":\"llama3\"}}";
            var cfg = LlamaConfigReader.Parse(Encoding.UTF8.GetBytes(json));

            Assert.NotNull(cfg.RopeScaling);
            Assert.Equal(32f, cfg.RopeScaling!.Factor);
            Assert.Equal(1f, cfg.RopeScaling.LowFreqFactor);
            Assert.Equal(4f, cfg.RopeScaling.HighFreqFactor);
            Assert.Equal(8192, cfg.RopeScaling.OriginalContextLength);
        }

        [Fact]
        public void Parse_NonLlama3RopeScaling_Ignored()
        {
            // "linear"/"dynamic"/"yarn" aren't implemented → fall back to plain RoPE (null).
            const string json =
                "{\"hidden_size\":64,\"num_attention_heads\":4,\"num_hidden_layers\":2,\"vocab_size\":100," +
                "\"rope_scaling\":{\"type\":\"linear\",\"factor\":4.0}}";
            var cfg = LlamaConfigReader.Parse(Encoding.UTF8.GetBytes(json));

            Assert.Null(cfg.RopeScaling);
            Assert.Equal(2, cfg.NLayers);   // rest of the config still parsed (object skipped cleanly)
        }
    }
}
