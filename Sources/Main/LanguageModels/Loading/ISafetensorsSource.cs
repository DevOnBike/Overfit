// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// A source of named safetensors weights — either a single file
    /// (<see cref="SafetensorsReader"/>) or a multi-file HuggingFace shard set
    /// (<see cref="ShardedSafetensorsReader"/>). Loaders depend on this so they work
    /// with both without caring how the bytes are split across files.
    /// </summary>
    public interface ISafetensorsSource : IDisposable
    {
        /// <summary>Tensor name → dtype / shape / byte range (across all shards).</summary>
        IReadOnlyDictionary<string, SafetensorsTensorInfo> Tensors
        {
            get;
        }

        /// <summary>Element count (product of shape dims) for a named tensor.</summary>
        long ElementCount(string name);

        /// <summary>Reads <paramref name="name"/> and dequantizes it to F32 into <paramref name="destination"/>.</summary>
        void LoadF32(string name, Span<float> destination);
    }
}
