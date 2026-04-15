// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Data;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Data.Prepare;
using DevOnBike.Overfit.Data.Tabular;
using DevOnBike.Overfit.Tensors;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Prepare
{
    public class DataPreparationTests
    {
        private readonly ITestOutputHelper _output;

        public DataPreparationTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void FullPipeline_Should_Convert_Clean_And_Execute_Successfully()
        {
            // 1. DANE TESTOWE (Makieta z Excela)
            var rawRows = new List<PropertyData>
            {
                new()
                {
                    Powierzchnia = 50,
                    Pietro = 2,
                    CzyKamienica = false,
                    CzyMaKomorke = true,
                    PowKomorki = 4,
                    Miasto = "Warszawa",
                    NazwaAgencji = "Premium",
                    Cena = 600000
                },
                new()
                {
                    Powierzchnia = 60,
                    Pietro = 3,
                    CzyKamienica = true,
                    CzyMaKomorke = false,
                    PowKomorki = 0,
                    Miasto = "Krakow",
                    NazwaAgencji = "Standard",
                    Cena = 500000
                }
            };

            var schema = new TableSchema
            {
                Features = new List<ColumnDefinition>
                {
                    new() { Name = nameof(PropertyData.Powierzchnia), Type = ColumnType.Numeric },
                    new() { Name = nameof(PropertyData.Pietro), Type = ColumnType.Numeric },
                    new() { Name = nameof(PropertyData.CzyKamienica), Type = ColumnType.Binary },
                    new() { Name = nameof(PropertyData.CzyMaKomorke), Type = ColumnType.Binary },
                    new() { Name = nameof(PropertyData.PowKomorki), Type = ColumnType.Numeric },
                    new() { Name = nameof(PropertyData.Miasto), Type = ColumnType.Categorical },
                    new() { Name = nameof(PropertyData.NazwaAgencji), Type = ColumnType.Categorical }
                },
                Target = new ColumnDefinition { Name = nameof(PropertyData.Cena), Type = ColumnType.Numeric }
            };

            var converter = new TabularToTensorConverter<PropertyData>(schema, (item, propName) => item.GetType().GetProperty(propName)?.GetValue(item));
            converter.Fit(rawRows);
            var (features, targets) = converter.Convert(rawRows);

            var pipeline = new DataPipeline()
                .AddLayer(new TechnicalSanityLayer())
                .AddLayer(new RobustScalingLayer());

            var ctx = new PipelineContext(features, targets);
            var result = pipeline.Execute(features, targets);

            Assert.NotNull(result);
            Assert.True(result.Features.GetView().GetDim(0) > 0);
        }

        [Fact]
        public void RobustScalingLayer_Should_Scale_Correctly()
        {
            var features = new FastTensor<float>(3, 3, clearMemory: true);
            var targets = new FastTensor<float>(3, 1, clearMemory: true);
            var fView = features.GetView();

            // Kolumna 0
            fView[0, 0] = 10f; fView[1, 0] = 20f; fView[2, 0] = 30f;
            // Kolumna 1
            fView[0, 1] = 100f; fView[1, 1] = 200f; fView[2, 1] = 300f;
            // Kolumna 2
            fView[0, 2] = 500000f; fView[1, 2] = 500000f; fView[2, 2] = 500000f;

            var ctx = new PipelineContext(features, targets);
            var scaler = new RobustScalingLayer();

            var result = scaler.Process(ctx);
            var resView = result.Features.GetView();

            // (10 - 20) / 10 = -1
            Assert.Equal(-1f, resView[0, 0]);
            Assert.Equal(0f, resView[1, 0]);
            Assert.Equal(1f, resView[2, 0]);

            // (100 - 200) / 100 = -1
            Assert.Equal(-1f, resView[0, 1]);
            Assert.Equal(0f, resView[1, 1]);
            Assert.Equal(1f, resView[2, 1]);

            // Mediana kolumny 2 = 500000 → (500000-500000)/IQR = 0
            Assert.Equal(0f, resView[0, 2]);
            Assert.Equal(0f, resView[1, 2]);
            Assert.Equal(0f, resView[2, 2]);
        }

        [Fact]
        public void CorrelationFilter_Should_Remove_Redundant_Columns()
        {
            // 5 wierszy, 3 kolumny — kolumna 0 i 2 niemal identyczne (r ≈ 1.0)
            var features = new FastTensor<float>(5, 3, clearMemory: true);
            var targets = new FastTensor<float>(5, 1, clearMemory: true);
            var fSpan = features.GetView().AsSpan();
            var tSpan = targets.GetView().AsSpan();

            // Kolumna 0: [1, 2, 3, 4, 5]
            // Kolumna 1: [10, 5, 20, 3, 15] — nieskorelowana (nieliniowa, chaotyczna)
            // Kolumna 2: [1.01, 2.02, 3.03, 4.04, 5.05] — niemal kopia kolumny 0
            float[] col0 = [1f, 2f, 3f, 4f, 5f];
            float[] col1 = [10f, 5f, 20f, 3f, 15f];
            float[] col2 = [1.01f, 2.02f, 3.03f, 4.04f, 5.05f];
            float[] targ = [1f, 2f, 3f, 4f, 5f]; // Targets mocno skorelowane z kolumną 0

            for (var r = 0; r < 5; r++)
            {
                fSpan[r * 3 + 0] = col0[r];
                fSpan[r * 3 + 1] = col1[r];
                fSpan[r * 3 + 2] = col2[r];
                tSpan[r] = targ[r];
            }

            var ctx = new PipelineContext(features, targets);
            var filter = new CorrelationFilterLayer(threshold: 0.95f, strategy: DropStrategy.KeepHigherTargetCorrelation);

            var result = filter.Process(ctx);

            // Spodziewamy się usunięcia jednej z redundantnych kolumn (zostaną 2)
            Assert.Equal(2, result.Features.GetView().GetDim(1));
        }

        // Mock data structure for testing
        private class PropertyData
        {
            public float Powierzchnia { get; set; }
            public float Pietro { get; set; }
            public bool CzyKamienica { get; set; }
            public bool CzyMaKomorke { get; set; }
            public float PowKomorki { get; set; }
            public string Miasto { get; set; }
            public string NazwaAgencji { get; set; }
            public float Cena { get; set; }
        }
    }
}