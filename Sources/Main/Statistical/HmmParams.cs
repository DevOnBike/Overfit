// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Tensors;

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
        /// Konwertuje płaskie tablice z JSON-a na zoptymalizowane FastTensor dla silnika Overfit.
        /// Zwraca tensory, które po przekazaniu do GaussianHMM zostaną tam zutylizowane (Dispose).
        /// </summary>
        public FastTensor<float>[] ToFastTensors(int featureCount)
        {
            var stateCount = Covariances.Length;
            var tensors = new FastTensor<float>[stateCount];

            for (var i = 0; i < stateCount; i++)
            {
                tensors[i] = new FastTensor<float>(featureCount, featureCount, clearMemory: false);
                Covariances[i].CopyTo(tensors[i].GetView().AsSpan());
            }

            return tensors;
        }
    }
}