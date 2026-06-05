// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Adapters
{
    public class CachedGpt1ModelAdapterTests
    {
        [Fact]
        public void Constructor_ExposesModelShape()
        {
            using var model = CreateSmallModel();
            using var adapter = new CachedGpt1ModelAdapter(model);

            Assert.Equal(model.Config.NLayers, adapter.LayerCount);
            Assert.Equal(model.Config.DModel, adapter.DModel);
            Assert.Equal(model.Config.NHeads, adapter.HeadCount);
            Assert.Equal(model.Config.DModel / model.Config.NHeads, adapter.HeadDimension);
            Assert.Equal(model.Config.DFF, adapter.DFF);
            Assert.Equal(model.Config.VocabSize, adapter.VocabSize);
            Assert.Equal(model.Config.ContextLength, adapter.MaxContextLength);
            Assert.Equal(0, adapter.CurrentPosition);
            Assert.False(adapter.IsFull);
        }

        [Fact]
        public void DecodeNextToken_AdvancesPositionAndWritesFiniteLogits()
        {
            using var model = CreateSmallModel();
            using var adapter = new CachedGpt1ModelAdapter(model);

            var logits = new float[model.Config.VocabSize];

            adapter.DecodeNextToken(
                tokenId: 1,
                logits);

            Assert.Equal(1, adapter.CurrentPosition);
            Assert.DoesNotContain(logits, float.IsNaN);
            Assert.DoesNotContain(logits, float.IsInfinity);
        }

        [Fact]
        public void DecodeNextToken_TwoTokens_AdvancesPositionTwice()
        {
            using var model = CreateSmallModel();
            using var adapter = new CachedGpt1ModelAdapter(model);

            var logits = new float[model.Config.VocabSize];

            adapter.DecodeNextToken(1, logits);
            adapter.DecodeNextToken(2, logits);

            Assert.Equal(2, adapter.CurrentPosition);
        }

        [Fact]
        public void Reset_ClearsPositionAndLastLogits()
        {
            using var model = CreateSmallModel();
            using var adapter = new CachedGpt1ModelAdapter(model);

            var logits = new float[model.Config.VocabSize];

            adapter.DecodeNextToken(1, logits);

            Assert.Equal(1, adapter.CurrentPosition);

            adapter.Reset();

            Assert.Equal(0, adapter.CurrentPosition);

            var lastLogits = new float[model.Config.VocabSize];
            adapter.GetLastLogits(lastLogits);

            Assert.All(lastLogits, value => Assert.Equal(0f, value));
        }

        [Fact]
        public void GetLastLogits_ReturnsLastDecodedLogits()
        {
            using var model = CreateSmallModel();
            using var adapter = new CachedGpt1ModelAdapter(model);

            var logits = new float[model.Config.VocabSize];
            var copied = new float[model.Config.VocabSize];

            adapter.DecodeNextToken(1, logits);
            adapter.GetLastLogits(copied);

            Assert.Equal(logits, copied);
        }

        [Fact]
        public void DecodeNextToken_LogitsDestinationTooSmall_Throws()
        {
            using var model = CreateSmallModel();
            using var adapter = new CachedGpt1ModelAdapter(model);

            Assert.Throws<ArgumentException>(() =>
                adapter.DecodeNextToken(
                    tokenId: 1,
                    logits: new float[model.Config.VocabSize - 1]));
        }

        [Fact]
        public void DecodeNextToken_InvalidTokenId_Throws()
        {
            using var model = CreateSmallModel();
            using var adapter = new CachedGpt1ModelAdapter(model);

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                adapter.DecodeNextToken(
                    tokenId: model.Config.VocabSize,
                    logits: new float[model.Config.VocabSize]));
        }

        [Fact]
        public void DecodeNextToken_WhenCacheFull_Throws()
        {
            var config = new GPT1Config
            {
                VocabSize = 16,
                ContextLength = 2,
                DModel = 8,
                NHeads = 2,
                NLayers = 1,
                DFF = 16,
                TieWeights = true,
                PreLayerNorm = true
            };

            using var model = new GPT1Model(config);
            using var adapter = new CachedGpt1ModelAdapter(model);

            var logits = new float[config.VocabSize];

            adapter.DecodeNextToken(1, logits);
            adapter.DecodeNextToken(2, logits);

            Assert.True(adapter.IsFull);

            Assert.Throws<OverfitRuntimeException>(() =>
                adapter.DecodeNextToken(3, logits));
        }

        [Fact]
        public void RefreshWeightsFromModel_UsesUntiedLmHeadWeights()
        {
            var config = new GPT1Config
            {
                VocabSize = 16,
                ContextLength = 4,
                DModel = 8,
                NHeads = 2,
                NLayers = 1,
                DFF = 16,
                TieWeights = false,
                PreLayerNorm = true
            };

            using var model = new GPT1Model(config);
            using var adapter = new CachedGpt1ModelAdapter(model);

            model.LMHead.DataSpan.Clear();

            adapter.RefreshWeightsFromModel();

            var logits = new float[config.VocabSize];

            adapter.DecodeNextToken(
                tokenId: 1,
                logits);

            Assert.All(logits, value => Assert.Equal(0f, value));
        }

        [Fact]
        public void Dispose_ThenUse_Throws()
        {
            using var model = CreateSmallModel();
            var adapter = new CachedGpt1ModelAdapter(model);

            adapter.Dispose();

            Assert.Throws<ObjectDisposedException>(() =>
                adapter.Reset());

            Assert.Throws<ObjectDisposedException>(() =>
                adapter.DecodeNextToken(
                    tokenId: 1,
                    logits: new float[model.Config.VocabSize]));
        }

        private static GPT1Model CreateSmallModel()
        {
            var config = new GPT1Config
            {
                VocabSize = 32,
                ContextLength = 8,
                DModel = 16,
                NHeads = 2,
                NLayers = 1,
                DFF = 32,
                TieWeights = true,
                PreLayerNorm = true
            };

            var model = new GPT1Model(config);
            model.Eval();

            return model;
        }
    }
}
