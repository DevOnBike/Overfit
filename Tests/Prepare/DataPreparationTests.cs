// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Data.Prepare;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Prepare
{
    public class DataPreparationTests
    {
        private readonly ITestOutputHelper _output;

        public DataPreparationTests(ITestOutputHelper output) => _output = output;

        [Fact]
        public void FullPipeline_Should_Convert_Clean_And_Execute_Successfully()
        {
            // 1. DANE TESTOWE (Makieta z Excela)
            var rawRows = new List<PropertyData>
            {
                new() { Powierzchnia = 50, Pietro = 2, CzyKamienica = false, CzyMaKomorke = true,  PowKomorki = 4,  Miasto = "Warszawa", NazwaAgencji = "Premium",  Cena = 600000 },
                new() { Powierzchnia = 30, Pietro = 0, CzyKamienica = true,  CzyMaKomorke = false, PowKomorki = 0,  Miasto = "Krakow",   NazwaAgencji = "Standard", Cena = 400000 },
                new() { Powierzchnia = 80, Pietro = 5, CzyKamienica = false, CzyMaKomorke = true,  PowKomorki = 10, Miasto = "Warszawa", NazwaAgencji = "Standard", Cena = 950000 },
            };

            // 2. DEFINICJA SCHEMATU
            var schema = new TableSchema
            {
                Features =
                [
                    new() { Name = "Powierzchnia", Type = ColumnType.Numeric },    // Idx 0
                    new() { Name = "Pietro",       Type = ColumnType.Numeric },    // Idx 1
                    new() { Name = "CzyKamienica", Type = ColumnType.Binary },     // Idx 2
                    new() { Name = "CzyMaKomorke", Type = ColumnType.Binary },     // Idx 3
                    new() { Name = "PowKomorki",   Type = ColumnType.Numeric },    // Idx 4
                    new() { Name = "Miasto",       Type = ColumnType.Categorical },// Idx 5, 6 (Krakow, Warszawa)
                    new() { Name = "NazwaAgencji", Type = ColumnType.Categorical },// Idx 7, 8 (Premium, Standard)
                ],
                Target = new ColumnDefinition { Name = "Cena", Type = ColumnType.Numeric }
            };

            // 3. KONWERSJA (Excel -> FastTensor)
            // ZMIANA: Wstrzykujemy AOT-Safe delegat zamiast Refleksji
            var converter = new TabularToTensorConverter<PropertyData>(schema, (item, propName) => propName switch
            {
                "Powierzchnia" => item.Powierzchnia,
                "Pietro" => item.Pietro,
                "CzyKamienica" => item.CzyKamienica,
                "CzyMaKomorke" => item.CzyMaKomorke,
                "PowKomorki" => item.PowKomorki,
                "Miasto" => item.Miasto,
                "NazwaAgencji" => item.NazwaAgencji,
                "Cena" => item.Cena,
                _ => null
            });

            converter.Fit(rawRows);
            var (rawX, rawY) = converter.Transform(rawRows);

            // 4. PRZYGOTOWANIE PIPELINE
            // Indeksy kolumn binarnych/one-hot — nie podlegają skalowaniu ani winsoryzacji
            var binaryAndOneHot = new HashSet<int> { 2, 3, 5, 6, 7, 8 };
            var numericColumns = new HashSet<int> { 0, 1, 4 };

            var pipeline = new DataPipeline()
                .AddLayer(new TechnicalSanityLayer())                               // Czyści NaN/Inf/Subnormale
                .AddLayer(new DuplicateRowFilterLayer())                             // Wyrzuca zduplikowane wiersze
                .AddLayer(new ConstantColumnFilterLayer())                           // Wyrzuca kolumny o zerowej wariancji
                .AddLayer(new OutlierClipLayer(                                      // Winsoryzacja — tylko kolumny numeryczne
                    excludedColumns: binaryAndOneHot))
                .AddLayer(new RobustScalingLayer(                                    // Skalowanie IQR — tylko kolumny numeryczne
                    columnIndices: numericColumns,
                    excludedColumns: binaryAndOneHot));

            // 5. EGZEKUCJA POTOKU
            using var finalContext = pipeline.Execute(rawX, rawY);

            // 6. WERYFIKACJA (ASSERT)
            // Kształt: 5 (podst.) + 2 (miasta) + 2 (agencje) = 9
            // Wszystkie kolumny mają wariancję w 3 wierszach → ConstantColumnFilter nic nie wyrzuca
            Assert.Equal(3, finalContext.Features.GetDim(0));
            Assert.Equal(9, finalContext.Features.GetDim(1));

            // Weryfikacja One-Hot Encodingu (Wiersz 0 to Warszawa)
            // Sortowanie alfabetyczne w Fit(): [5] = Krakow, [6] = Warszawa
            // RobustScaling pominął te kolumny dzięki excludedColumns
            Assert.Equal(0f, finalContext.Features[0, 5]); // Krakow
            Assert.Equal(1f, finalContext.Features[0, 6]); // Warszawa

            // Sprawdzenie skalowania numerycznego (Powierzchnia 50m przy zestawie [30, 50, 80])
            // Mediana = 50, więc (50-50)/IQR = 0
            Assert.Equal(0f, finalContext.Features[0, 0]);

            // Kolumny binarne nietknięte
            Assert.Equal(0f, finalContext.Features[0, 2]); // CzyKamienica = false → 0
            Assert.Equal(1f, finalContext.Features[0, 3]); // CzyMaKomorke = true  → 1

            _output.WriteLine("Pipeline przetworzył dane pomyślnie.");
            _output.WriteLine($"Finalny kształt: {finalContext.Features.GetDim(0)}x{finalContext.Features.GetDim(1)}");
        }

        [Fact]
        public void TechnicalSanityLayer_Should_Remove_Rows_With_Excessive_NaN()
        {
            // 4 wiersze, 3 kolumny — wiersz 2 ma 100% NaN
            var features = new FastTensor<float>(4, 3);
            var targets = new FastTensor<float>(4, 1);

            var fSpan = features.AsSpan();
            fSpan[0] = 1f; fSpan[1] = 2f; fSpan[2] = 3f;            // Wiersz 0: czysty
            fSpan[3] = 4f; fSpan[4] = float.NaN; fSpan[5] = 5f;     // Wiersz 1: 1/3 NaN → OK
            fSpan[6] = float.NaN; fSpan[7] = float.NaN; fSpan[8] = float.NaN; // Wiersz 2: 3/3 NaN → wyrzucony
            fSpan[9] = 7f; fSpan[10] = 8f; fSpan[11] = 9f;          // Wiersz 3: czysty

            targets.AsSpan()[0] = 10f;
            targets.AsSpan()[1] = 20f;
            targets.AsSpan()[2] = 30f;
            targets.AsSpan()[3] = 40f;

            // Próg 50% — wiersz z >50% NaN zostanie wyrzucony
            var layer = new TechnicalSanityLayer(maxCorruptedRatio: 0.5f);
            using var result = layer.Process(new PipelineContext(features, targets));

            Assert.Equal(3, result.Features.GetDim(0)); // Wiersz 2 wyrzucony
            Assert.Equal(3, result.Features.GetDim(1)); // Kolumny bez zmian

            // NaN w wierszu 1 (który przeszedł) został zastąpiony zerem
            Assert.Equal(0f, result.Features[1, 1]);

            // Target wiersza 3 (przesunięty na pozycję 2) zachowany
            Assert.Equal(40f, result.Targets[2, 0]);
        }

        [Fact]
        public void DuplicateRowFilter_Should_Remove_Exact_Duplicates()
        {
            // 4 wiersze, 2 kolumny — wiersz 0 i 2 identyczne
            var features = new FastTensor<float>(4, 2);
            var targets = new FastTensor<float>(4, 1);

            var fSpan = features.AsSpan();
            fSpan[0] = 1f; fSpan[1] = 2f; // Wiersz 0
            fSpan[2] = 3f; fSpan[3] = 4f; // Wiersz 1 — unikat
            fSpan[4] = 1f; fSpan[5] = 2f; // Wiersz 2 — duplikat wiersza 0
            fSpan[6] = 5f; fSpan[7] = 6f; // Wiersz 3 — unikat

            targets.AsSpan()[0] = 10f;
            targets.AsSpan()[1] = 20f;
            targets.AsSpan()[2] = 30f; // Inny Target, ale Features identyczne
            targets.AsSpan()[3] = 40f;

            var layer = new DuplicateRowFilterLayer(includeTargetInComparison: false);
            using var result = layer.Process(new PipelineContext(features, targets));

            Assert.Equal(3, result.Features.GetDim(0)); // Wiersz 2 wyrzucony
            Assert.Equal(10f, result.Targets[0, 0]);    // Zachowany pierwszy z pary
            Assert.Equal(20f, result.Targets[1, 0]);
            Assert.Equal(40f, result.Targets[2, 0]);
        }

        [Fact]
        public void DuplicateRowFilter_IncludeTarget_Should_Keep_Different_Targets()
        {
            var features = new FastTensor<float>(3, 2);
            var targets = new FastTensor<float>(3, 1);

            var fSpan = features.AsSpan();
            fSpan[0] = 1f; fSpan[1] = 2f;
            fSpan[2] = 1f; fSpan[3] = 2f; // Te same Features, inny Target
            fSpan[4] = 1f; fSpan[5] = 2f; // Te same Features, jeszcze inny Target

            targets.AsSpan()[0] = 100f;
            targets.AsSpan()[1] = 200f;
            targets.AsSpan()[2] = 100f; // Ten sam Target co wiersz 0 → duplikat

            var layer = new DuplicateRowFilterLayer(includeTargetInComparison: true);
            using var result = layer.Process(new PipelineContext(features, targets));

            // Wiersz 0 i 1 mają różne Targety → oba zachowane
            // Wiersz 2 = duplikat wiersza 0 (Features + Target identyczne) → wyrzucony
            Assert.Equal(2, result.Features.GetDim(0));
        }

        [Fact]
        public void ConstantColumnFilter_Should_Remove_Zero_Variance_Columns()
        {
            // 4 wiersze, 3 kolumny — kolumna 1 stała
            var features = new FastTensor<float>(4, 3);
            var targets = new FastTensor<float>(4, 1);

            var fSpan = features.AsSpan();
            fSpan[0] = 1f; fSpan[1] = 5f; fSpan[2] = 10f;
            fSpan[3] = 2f; fSpan[4] = 5f; fSpan[5] = 20f;
            fSpan[6] = 3f; fSpan[7] = 5f; fSpan[8] = 30f;
            fSpan[9] = 4f; fSpan[10] = 5f; fSpan[11] = 40f;

            var layer = new ConstantColumnFilterLayer();
            using var result = layer.Process(new PipelineContext(features, targets));

            Assert.Equal(4, result.Features.GetDim(0)); // Wiersze bez zmian
            Assert.Equal(2, result.Features.GetDim(1)); // Kolumna 1 (stała=5) wyrzucona

            // Kolumna 0 i 2 zachowane, przenumerowane na 0 i 1
            Assert.Equal(1f, result.Features[0, 0]);
            Assert.Equal(10f, result.Features[0, 1]);
        }

        [Fact]
        public void OutlierClipLayer_Should_Clip_Extreme_Values()
        {
            // 10 wierszy, 1 kolumna — z jednym outlierem
            var features = new FastTensor<float>(10, 1);
            var targets = new FastTensor<float>(10, 1);
            var fSpan = features.AsSpan();

            for (var i = 0; i < 9; i++)
            {
                fSpan[i] = i + 1; // 1..9
            }
            fSpan[9] = 1000f; // Outlier

            // Agresywna winsoryzacja: 10%/90%
            var layer = new OutlierClipLayer(
                lowerPercentile: 0.1f,
                upperPercentile: 0.9f);

            using var result = layer.Process(new PipelineContext(features, targets));

            // Outlier 1000 przycięty do wartości 90. percentyla
            var clipped = result.Features[9, 0];
            Assert.True(clipped < 1000f, $"Outlier powinien być przycięty, ale wynosi {clipped}");
            Assert.True(clipped >= 1f, $"Przycięta wartość {clipped} nie powinna być poniżej minimum danych");
        }

        [Fact]
        public void LogTransformLayer_Log1p_Should_Compress_Skewed_Distribution()
        {
            // Typowy rozkład cen: 200k, 400k, 600k, 2M (skośny prawy ogon)
            var features = new FastTensor<float>(4, 2);
            var targets = new FastTensor<float>(4, 1);
            var fSpan = features.AsSpan();

            // Kolumna 0: cena (skośna), Kolumna 1: piętro (symetryczna — nie transformujemy)
            fSpan[0] = 200000f; fSpan[1] = 1f;
            fSpan[2] = 400000f; fSpan[3] = 3f;
            fSpan[4] = 600000f; fSpan[5] = 5f;
            fSpan[6] = 2000000f; fSpan[7] = 2f;

            var layer = new LogTransformLayer(
                columnIndices: new List<int> { 0 },
                mode: LogMode.Log1p);

            using var result = layer.Process(new PipelineContext(features, targets));

            // Po log1p rozrzut powinien być drastycznie mniejszy
            var transformedMin = result.Features[0, 0]; // log(1 + 200000)
            var transformedMax = result.Features[3, 0]; // log(1 + 2000000)

            // Przed: max/min = 2000000/200000 = 10x
            // Po:    max/min ≈ 14.5/12.2 ≈ 1.19x
            var ratio = transformedMax / transformedMin;
            Assert.True(ratio < 2f, $"Log1p powinien skompresować rozkład, ratio = {ratio:F2}");

            // Kolumna 1 (piętro) nietknięta
            Assert.Equal(1f, result.Features[0, 1]);
            Assert.Equal(3f, result.Features[1, 1]);
        }

        [Fact]
        public void LogTransformLayer_SignedLog1p_Should_Preserve_Sign()
        {
            // Zmiana ceny r/r: -50000, +20000, -100000, +500000
            var features = new FastTensor<float>(4, 1);
            var targets = new FastTensor<float>(4, 1);
            var fSpan = features.AsSpan();

            fSpan[0] = -50000f;
            fSpan[1] = 20000f;
            fSpan[2] = -100000f;
            fSpan[3] = 500000f;

            var layer = new LogTransformLayer(
                columnIndices: new List<int> { 0 },
                mode: LogMode.SignedLog1p);

            using var result = layer.Process(new PipelineContext(features, targets));

            // Ujemne wartości zachowują znak
            Assert.True(result.Features[0, 0] < 0f, "Ujemna zmiana ceny powinna pozostać ujemna");
            Assert.True(result.Features[2, 0] < 0f, "Ujemna zmiana ceny powinna pozostać ujemna");

            // Dodatnie wartości zachowują znak
            Assert.True(result.Features[1, 0] > 0f, "Dodatnia zmiana ceny powinna pozostać dodatnia");
            Assert.True(result.Features[3, 0] > 0f, "Dodatnia zmiana ceny powinna pozostać dodatnia");

            // |log(-100k)| > |log(-50k)| — zachowany porządek bezwzględny
            Assert.True(MathF.Abs(result.Features[2, 0]) > MathF.Abs(result.Features[0, 0]));
        }

        [Fact]
        public void RobustScalingLayer_Should_Scale_Only_Specified_Columns()
        {
            // 5 wierszy, 3 kolumny: [numeryczna, binarna, numeryczna]
            var features = new FastTensor<float>(5, 3);
            var targets = new FastTensor<float>(5, 1);
            var fSpan = features.AsSpan();

            // Kolumna 0: powierzchnia [30, 40, 50, 60, 80]
            // Kolumna 1: binarna [0, 1, 0, 1, 0]
            // Kolumna 2: cena [300k, 400k, 500k, 600k, 800k]
            fSpan[0] = 30f; fSpan[1] = 0f; fSpan[2] = 300000f;
            fSpan[3] = 40f; fSpan[4] = 1f; fSpan[5] = 400000f;
            fSpan[6] = 50f; fSpan[7] = 0f; fSpan[8] = 500000f;
            fSpan[9] = 60f; fSpan[10] = 1f; fSpan[11] = 600000f;
            fSpan[12] = 80f; fSpan[13] = 0f; fSpan[14] = 800000f;

            var layer = new RobustScalingLayer(
                columnIndices: new HashSet<int> { 0, 2 },
                excludedColumns: new HashSet<int> { 1 });

            using var result = layer.Process(new PipelineContext(features, targets));

            // Mediana kolumny 0 = 50 → (50-50)/IQR = 0
            Assert.Equal(0f, result.Features[2, 0]);

            // Kolumna 1 (binarna) nietknięta
            Assert.Equal(0f, result.Features[0, 1]);
            Assert.Equal(1f, result.Features[1, 1]);

            // Mediana kolumny 2 = 500000 → (500000-500000)/IQR = 0
            Assert.Equal(0f, result.Features[2, 2]);
        }

        [Fact]
        public void CorrelationFilter_Should_Remove_Redundant_Columns()
        {
            // 5 wierszy, 3 kolumny — kolumna 0 i 2 niemal identyczne (r ≈ 1.0)
            var features = new FastTensor<float>(5, 3);
            var targets = new FastTensor<float>(5, 1);
            var fSpan = features.AsSpan();
            var tSpan = targets.AsSpan();

            // Kolumna 0: [1, 2, 3, 4, 5]
            // Kolumna 1: [10, 5, 20, 3, 15] — nieskorelowana (nieliniowa, chaotyczna)
            // Kolumna 2: [1.01, 2.02, 3.03, 4.04, 5.05] — niemal kopia kolumny 0
            float[] col1 = [10f, 5f, 20f, 3f, 15f];

            for (var r = 0; r < 5; r++)
            {
                fSpan[r * 3 + 0] = r + 1;
                fSpan[r * 3 + 1] = col1[r];
                fSpan[r * 3 + 2] = (r + 1) * 1.01f;
                tSpan[r] = (r + 1) * 100f;
            }

            var layer = new CorrelationFilterLayer(threshold: 0.99f);
            using var result = layer.Process(new PipelineContext(features, targets));

            // Para (0, 2) skorelowana → jedna wyrzucona → zostają 2 kolumny
            Assert.Equal(2, result.Features.GetDim(1));
            Assert.Equal(5, result.Features.GetDim(0));
        }
    }
}