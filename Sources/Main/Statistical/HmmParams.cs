// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Data.Serialization;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistical
{
    public class HmmParams
    {
        public required float[] InitialProbs { get; init; }
        public required float[] TransitionMatrix { get; init; }
        public required float[] Means { get; init; }

        /// <summary>
        /// All per-state covariance matrices concatenated into ONE flat row-major buffer (state-major):
        /// <c>stateCount</c> blocks of <c>featureCount × featureCount</c>. Flat (not jagged <c>float[][]</c>)
        /// to stay consistent with the rest of the engine — one allocation, cache-friendly; the JSON is a
        /// single flat array. <see cref="ToFastTensors"/> splits it back into per-state matrices.
        /// </summary>
        public required float[] Covariances { get; init; }

        public void SaveToFile(string path)
        {
            var json = JsonSerializer.Serialize(this, OverfitJsonContext.Default.HmmParams);
            File.WriteAllText(path, json);
        }

        public static HmmParams LoadFromFile(string path)
        {
            var json = File.ReadAllText(path);

            return JsonSerializer.Deserialize(json, OverfitJsonContext.Default.HmmParams) ?? throw new InvalidOperationException("Nie udało się zdeserializować HmmParams.");
        }

        /// <summary>
        /// Converts flat arrays from JSON into optimized FastTensors for the Overfit engine.
        /// Returns tensors that will be disposed by GaussianHMM after being passed to it.
        /// </summary>
        public FastTensor<float>[] ToFastTensors(int featureCount)
        {
            var matSize = featureCount * featureCount;
            var stateCount = Covariances.Length / matSize;
            var tensors = new FastTensor<float>[stateCount];

            for (var i = 0; i < stateCount; i++)
            {
                tensors[i] = new FastTensor<float>(featureCount, featureCount, clearMemory: false);
                Covariances.AsSpan(i * matSize, matSize).CopyTo(tensors[i].GetView().AsSpan());
            }

            return tensors;
        }
    }
}