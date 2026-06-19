// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// Reads a HuggingFace sharded safetensors model: a <c>model.safetensors.index.json</c>
    /// whose <c>weight_map</c> maps each tensor to one of several
    /// <c>model-0000k-of-0000n.safetensors</c> shards. Each shard is opened as a
    /// <see cref="SafetensorsReader"/> and weights are read on demand from the owning
    /// shard — no shard is fully materialised in memory.
    /// </summary>
    public sealed class ShardedSafetensorsReader : ISafetensorsSource
    {
        private readonly SafetensorsReader[] _shards;
        private readonly Dictionary<string, SafetensorsReader> _tensorToShard;
        private bool _disposed;

        public ShardedSafetensorsReader(string indexPath)
        {
            if (!File.Exists(indexPath))
            {
                throw new FileNotFoundException($"Shard index '{indexPath}' not found.", indexPath);
            }

            var baseDir = Path.GetDirectoryName(Path.GetFullPath(indexPath)) ?? ".";
            var weightMap = ParseWeightMap(File.ReadAllBytes(indexPath));
            if (weightMap.Count == 0)
            {
                throw new OverfitFormatException($"Shard index '{indexPath}' has an empty weight_map.");
            }

            // Open each distinct shard once.
            var shardByFile = new Dictionary<string, SafetensorsReader>(StringComparer.Ordinal);
            var tensors = new Dictionary<string, SafetensorsTensorInfo>(weightMap.Count);
            var tensorToShard = new Dictionary<string, SafetensorsReader>(weightMap.Count);

            try
            {
                foreach (var (tensorName, shardFile) in weightMap)
                {
                    if (!shardByFile.TryGetValue(shardFile, out var shard))
                    {
                        shard = new SafetensorsReader(Path.Combine(baseDir, shardFile));
                        shardByFile[shardFile] = shard;
                    }

                    if (!shard.Tensors.TryGetValue(tensorName, out var info))
                    {
                        throw new OverfitFormatException(
                            $"weight_map points '{tensorName}' at shard '{shardFile}', but that shard does not contain it.");
                    }

                    tensors[tensorName] = info;
                    tensorToShard[tensorName] = shard;
                }
            }
            catch
            {
                foreach (var s in shardByFile.Values)
                {
                    s.Dispose();
                }
                throw;
            }

            _shards = new SafetensorsReader[shardByFile.Count];
            shardByFile.Values.CopyTo(_shards, 0);
            _tensorToShard = tensorToShard;
            Tensors = tensors;
        }

        public IReadOnlyDictionary<string, SafetensorsTensorInfo> Tensors
        {
            get;
        }

        public int ShardCount => _shards.Length;

        public long ElementCount(string name)
        {
            if (!Tensors.TryGetValue(name, out var info))
            {
                throw new KeyNotFoundException($"Tensor '{name}' not found across shards.");
            }
            return info.ElementCount;
        }

        public void LoadF32(string name, Span<float> destination)
        {
            ThrowIfDisposed();
            if (!_tensorToShard.TryGetValue(name, out var shard))
            {
                throw new KeyNotFoundException($"Tensor '{name}' not found across shards.");
            }
            shard.LoadF32(name, destination);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            foreach (var s in _shards)
            {
                s.Dispose();
            }
        }

        // Parses { "metadata": {...}, "weight_map": { name: shardFile, ... } }.
        private static Dictionary<string, string> ParseWeightMap(ReadOnlySpan<byte> indexBytes)
        {
            var map = new Dictionary<string, string>();
            var reader = new Utf8JsonReader(indexBytes, isFinalBlock: true, state: default);

            if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            {
                throw new OverfitFormatException("Shard index is not a JSON object.");
            }

            while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
            {
                var prop = reader.GetString()!;
                reader.Read(); // onto the value

                if (prop != "weight_map")
                {
                    reader.Skip();
                    continue;
                }

                if (reader.TokenType != JsonTokenType.StartObject)
                {
                    throw new OverfitFormatException("weight_map is not a JSON object.");
                }
                while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
                {
                    var tensorName = reader.GetString()!;
                    reader.Read();
                    map[tensorName] = reader.GetString()!;
                }
            }

            return map;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(ShardedSafetensorsReader));
            }
        }
    }
}
