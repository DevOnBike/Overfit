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
                new() { Powierzchnia = 50, Pietro = 2, CzyKamienica = false, CzyMaKomorke = true, PowKomorki = 4, Miasto = "Warszawa", NazwaAgencji = "Premium", Cena = 600000 },
                new() { Powierzchnia = 30, Pietro = 0, CzyKamienica = true, CzyMaKomorke = false, PowKomorki = 0, Miasto = "Krakow", NazwaAgencji = "Standard", Cena = 400000 },
                new() { Powierzchnia = 80, Pietro = 5, CzyKamienica = false, CzyMaKomorke = true, PowKomorki = 10, Miasto = "Warszawa", NazwaAgencji = "Standard", Cena = 950000 }
            };

            // 2. DEFINICJA SCHEMATU
            var schema = new TableSchema
            {
                Features =
                [
                    new() { Name = "Powierzchnia", Type = ColumnType.Numeric }, // Idx 0
                    new() { Name = "Pietro", Type = ColumnType.Numeric },       // Idx 1
                    new() { Name = "CzyKamienica", Type = ColumnType.Binary },  // Idx 2
                    new() { Name = "CzyMaKomorke", Type = ColumnType.Binary },  // Idx 3
                    new() { Name = "PowKomorki", Type = ColumnType.Numeric },   // Idx 4
                    new() { Name = "Miasto", Type = ColumnType.Categorical },   // Idx 5, 6 (Krakow, Warszawa)
                    new() { Name = "NazwaAgencji", Type = ColumnType.Categorical } // Idx 7, 8
                ],
                Target = new ColumnDefinition { Name = "Cena", Type = ColumnType.Numeric }
            };

            // 3. KONWERSJA (Excel -> FastTensor)
            var converter = new TabularToTensorConverter<PropertyData>(schema);
            converter.Fit(rawRows); // Ustala szerokość i sortuje kategorie alfabetycznie
            var (rawX, rawY) = converter.Transform(rawRows);

            // 4. PRZYGOTOWANIE PIPELINE
            // Indeksy kolumn numerycznych wymagających skalowania: 0 (Pow), 1 (Pietro), 4 (PowKom)
            var numericIndices = new List<int> { 0, 1, 4 };

            var pipeline = new DataPipeline()
                .AddLayer(new TechnicalSanityLayer())        // Usuwa NaN/Inf
                .AddLayer(new AnomalyFilterLayer())          // Usuwa skrajne rekordy (MAD)
                .AddLayer(new RobustScalingLayer(numericIndices)); // Skaluje TYLKO liczby, pomija One-Hot

            // 5. EGZEKUCJA POTOKU
            // PipelineContext automatycznie zarządza Dispose() starych tensorów przy transformacjach
            using var finalContext = pipeline.Execute(rawX, rawY);

            // 6. WERYFIKACJA (ASSERT)
            // Sprawdzenie kształtu: 5 (podst.) + 2 (miasta) + 2 (agencje) = 9
            Assert.Equal(3, finalContext.Features.GetDim(0));
            Assert.Equal(9, finalContext.Features.GetDim(1));

            // Weryfikacja One-Hot Encodingu (Wiersz 0 to Warszawa)
            // Dzięki sortowaniu w Fit(): [5] = Krakow, [6] = Warszawa
            // RobustScaling pominął te kolumny, więc wartości zostają 0 i 1
            Assert.Equal(0f, finalContext.Features[0, 5]); // Krakow
            Assert.Equal(1f, finalContext.Features[0, 6]); // Warszawa

            // Sprawdzenie skalowania numerycznego (Powierzchnia 50m przy zestawie [30, 50, 80])
            // Mediana = 50, więc (50-50)/IQR = 0
            Assert.Equal(0f, finalContext.Features[0, 0]);

            _output.WriteLine("Pipeline przetworzył dane pomyślnie.");
            _output.WriteLine($"Finalny kształt: {finalContext.Features.GetDim(0)}x{finalContext.Features.GetDim(1)}");
        }
    }
}

