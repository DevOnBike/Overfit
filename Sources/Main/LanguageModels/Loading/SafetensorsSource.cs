// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.IO;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>Factory: open a single-file or sharded safetensors source from a path or directory.</summary>
    public static class SafetensorsSource
    {
        /// <summary>
        /// Opens a safetensors source:
        /// <list type="bullet">
        ///   <item>a <c>*.index.json</c> file → sharded;</item>
        ///   <item>a directory with <c>model.safetensors.index.json</c> → sharded;</item>
        ///   <item>a directory with <c>model.safetensors</c> → single file;</item>
        ///   <item>any other path → single file.</item>
        /// </list>
        /// </summary>
        public static ISafetensorsSource Open(string path)
        {
            if (string.IsNullOrEmpty(path)) { throw new ArgumentException("Path is empty.", nameof(path)); }

            if (path.EndsWith(".index.json", StringComparison.OrdinalIgnoreCase) && File.Exists(path))
            {
                return new ShardedSafetensorsReader(path);
            }

            if (Directory.Exists(path))
            {
                var index = Path.Combine(path, "model.safetensors.index.json");
                if (File.Exists(index)) { return new ShardedSafetensorsReader(index); }

                var single = Path.Combine(path, "model.safetensors");
                if (File.Exists(single)) { return new SafetensorsReader(single); }

                throw new FileNotFoundException(
                    $"No 'model.safetensors' or 'model.safetensors.index.json' in directory '{path}'.");
            }

            return new SafetensorsReader(path);
        }
    }
}
