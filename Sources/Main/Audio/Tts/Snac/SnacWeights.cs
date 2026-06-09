// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.Audio.Tts.Snac
{
    /// <summary>
    /// The SNAC decoder's weights, loaded F32 from the converted safetensors (see <c>Scripts/convert_snac.py</c>,
    /// which folds <c>weight_norm</c> into plain conv weights and names tensors to match this loader). Every tensor
    /// is held flat by name; the decoder knows the dims from <see cref="SnacConfig"/>. Reflection-free, so AOT-clean.
    /// </summary>
    internal sealed class SnacWeights
    {
        private readonly Dictionary<string, float[]> _tensors;

        private SnacWeights(Dictionary<string, float[]> tensors)
        {
            _tensors = tensors;
        }

        /// <summary>The flat F32 data of a named tensor.</summary>
        public float[] this[string name]
        {
            get
            {
                if (!_tensors.TryGetValue(name, out var data))
                {
                    throw new OverfitFormatException($"SNAC weights are missing tensor '{name}'.");
                }
                return data;
            }
        }

        public static SnacWeights Load(string safetensorsPath)
        {
            using var reader = new SafetensorsReader(safetensorsPath);
            var tensors = new Dictionary<string, float[]>(reader.Tensors.Count, StringComparer.Ordinal);
            foreach (var name in reader.Tensors.Keys)
            {
                var data = new float[reader.ElementCount(name)];
                reader.LoadF32(name, data);
                tensors[name] = data;
            }
            return new SnacWeights(tensors);
        }
    }
}
