// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Data.Tabular;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core; // Dodano DOD Core
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Prepare
{
    public class RealEstateFullCycleTests
    {
        private readonly ITestOutputHelper _output;

        public RealEstateFullCycleTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void Training_And_Prediction_Should_Work_EndToEnd()
        {
            var rawData = GenerateDummyData(500);
            var schema = CreatePropertySchema();

            var converter = new TabularToTensorConverter<PropertyData>(schema, valueExtractor: (item, propName) => propName switch
            {
                "Powierzchnia" => item.Powierzchnia,
                "Pietro" => item.Pietro,
                "CzyKamienica" => item.CzyKamienica,
                "CzyMaKomorke" => item.CzyMaKomorke,
                "PowKomorki" => item.PowKomorki,
                "Miasto" => item.Miasto,
                "NazwaAgencji" => item.NazwaAgencji,
                "Cena" => item.Cena,
                _ => throw new ArgumentException($"Unknown property: {propName}")
            });

            converter.Fit(rawData);

            // Konwerter wciąż zwraca FastTensor, robimy tymczasową przejściówkę do DOD
            var (oldFeatures, oldTargets) = converter.Convert(rawData);

            var nSamples = rawData.Count;
            var nFeatures = oldFeatures.Size / nSamples;

            using var features = new TensorStorage<float>(oldFeatures.Size, clearMemory: false);
            using var targets = new TensorStorage<float>(oldTargets.Size, clearMemory: false);

            oldFeatures.GetView().AsReadOnlySpan().CopyTo(features.AsSpan());
            oldTargets.GetView().AsReadOnlySpan().CopyTo(targets.AsSpan());

            var fSpan = features.AsSpan();
            for (var i = 0; i < nSamples; i++)
            {
                fSpan[i * nFeatures + 0] /= 100f;
                fSpan[i * nFeatures + 1] /= 10f;
            }

            var tSpan = targets.AsSpan();
            for (var i = 0; i < tSpan.Length; i++)
            {
                tSpan[i] /= 100000f;
            }

            var inputFeatures = nFeatures;

            using var X = new AutogradNode(features, new TensorShape(nSamples, nFeatures), false);
            using var Y = new AutogradNode(targets, new TensorShape(nSamples, 1), false);

            using var layer1 = new LinearLayer(inputFeatures, 32);
            using var layer2 = new LinearLayer(32, 16);
            using var layer3 = new LinearLayer(16, 1);

            var model = new Sequential(layer1, new ReluActivation(), layer2, new ReluActivation(), layer3);

            var adam = new Adam(model.Parameters(), 0.01f) { UseAdamW = true, WeightDecay = 0.0001f };

            var finalLoss = 0f;
            var graph = new ComputationGraph();

            for (var epoch = 0; epoch < 500; epoch++)
            {
                graph.Reset();
                adam.ZeroGrad();

                using var prediction = model.Forward(graph, X);
                using var loss = TensorMath.MSELoss(graph, prediction, Y);

                finalLoss = loss.DataView.AsReadOnlySpan()[0];

                graph.Backward(loss);
                adam.Step();
            }

            _output.WriteLine($"Final Real Estate Loss: {finalLoss:F4}");

            Assert.True(finalLoss < 2.0f, $"Model failed to learn patterns. Final loss: {finalLoss:F4}");
        }

        private List<PropertyData> GenerateDummyData(int count)
        {
            var rnd = new Random(42);
            var list = new List<PropertyData>(count);

            for (var i = 0; i < count; i++)
            {
                var pow = rnd.Next(30, 150);
                var pietro = rnd.Next(0, 10);
                var isKam = rnd.NextDouble() > 0.8;
                var miasto = rnd.NextDouble() > 0.5 ? "Warszawa" : "Krakow";

                var cena = pow * 10000f + (miasto == "Warszawa" ? 200000f : 0f) - pietro * 1000f;
                if (isKam)
                {
                    cena *= 1.2f;
                }

                list.Add(new PropertyData
                {
                    Powierzchnia = pow,
                    Pietro = pietro,
                    CzyKamienica = isKam,
                    CzyMaKomorke = rnd.NextDouble() > 0.5,
                    PowKomorki = rnd.Next(0, 10),
                    Miasto = miasto,
                    NazwaAgencji = "Agencja",
                    Cena = cena
                });
            }

            return list;
        }

        private TableSchema CreatePropertySchema()
        {
            return new TableSchema
            {
                Features = new List<ColumnDefinition>
                {
                    new ColumnDefinition { Name = "Powierzchnia", Type = ColumnType.Numeric },
                    new ColumnDefinition { Name = "Pietro", Type = ColumnType.Numeric },
                    new ColumnDefinition { Name = "CzyKamienica", Type = ColumnType.Binary },
                    new ColumnDefinition { Name = "CzyMaKomorke", Type = ColumnType.Binary },
                    new ColumnDefinition { Name = "PowKomorki", Type = ColumnType.Numeric },
                    new ColumnDefinition { Name = "Miasto", Type = ColumnType.Categorical },
                    new ColumnDefinition { Name = "NazwaAgencji", Type = ColumnType.Categorical }
                },
                Target = new ColumnDefinition { Name = "Cena", Type = ColumnType.Numeric }
            };
        }

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