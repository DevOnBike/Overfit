using System.Diagnostics;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class MnistInferenceTests
    {
        private const string ModelPrefix = "mnist_model_v1";
        private const string ImagesPath = "d:/ml/t10k-images.idx3-ubyte";
        private const string LabelsPath = "d:/ml/t10k-labels.idx1-ubyte";

        [Fact(Skip = "integration tests")]
        public void Model_LoadedFromDisk_ShouldClassifyUnseenDigitsCorrectly()
        {
            // --- ARRANGE ---
            // Ładujemy 100 zupełnie nowych obrazków (zbiór testowy t10k)
            var (testX, testY) = MnistLoader.Load(ImagesPath, LabelsPath, 100);
            
            // Inicjalizujemy predictor (ładuje wagi z plików .bin)
            using var predictor = new MnistPredictor(ModelPrefix);
            
            var correctPredictions = 0;
            var totalSamples = 100;

            // --- ACT ---
            var sw = Stopwatch.StartNew();
            for (var i = 0; i < totalSamples; i++)
            {
                // Wyciągamy wiersz jako tablicę (nasze wejście dla Predictora)
                var imagePixels = testX.Row(i).ToArray();
                var trueLabel = testY.ArgMax(i);

                // Odpytujemy AI
                var prediction = predictor.Predict(imagePixels);

                if (prediction == trueLabel)
                {
                    correctPredictions++;
                }
            }
            sw.Stop();

            var accuracy = (double)correctPredictions / totalSamples * 100;
            Debug.WriteLine($"Accuracy: {accuracy}% | Time per image: {sw.Elapsed.TotalMilliseconds / totalSamples:F4}ms");

            // --- ASSERT ---
            // Po 5 epokach treningu z Adamem, model POWINIEN mieć przynajmniej 85% skuteczności.
            // Jeśli ma mniej, oznacza to, że albo trening był za krótki, albo wagi źle się zapisały.
            Assert.True(accuracy >= 85.0, $"Skuteczność modelu jest zbyt niska: {accuracy}%");
        }
    }
}