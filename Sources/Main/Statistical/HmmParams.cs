// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Statistical
{
    public class HmmParams
    {
        public required float[] InitialProbs { get; init; }
        public required float[] TransitionMatrix { get; init; }
        public required float[] Means { get; init; }
        public required float[][] Covariances { get; init; } // Tablica tablic dla JSON-a

        public void SaveToFile(string path)
        {
            var json = JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(path, json);
        }

        public static HmmParams LoadFromFile(string path)
        {
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<HmmParams>(json) ?? throw new InvalidOperationException("Nie udało się zdeserializować HmmParams.");
        }

        /// <summary>
        /// Konwertuje płaskie tablice z JSON-a na zoptymalizowane FastMatrix dla Twojego silnika.
        /// Zwraca macierze, które po przekazaniu do GaussianHMM należy zwolnić (Dispose).
        /// </summary>
        public FastMatrix<float>[] ToFastMatrices(int featureCount)
        {
            int stateCount = Covariances.Length;
            var matrices = new FastMatrix<float>[stateCount];
            for (int i = 0; i < stateCount; i++)
            {
                matrices[i] = new FastMatrix<float>(featureCount, featureCount);
                matrices[i].CopyFrom(Covariances[i]);
            }
            return matrices;
        }
    }
}