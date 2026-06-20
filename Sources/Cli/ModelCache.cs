// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Cli
{
    /// <summary>The local model store — <c>~/.overfit/models</c> — and resolution of a user-given model
    /// name (or path) to a GGUF file.</summary>
    internal static class ModelCache
    {
        /// <summary>The models directory (created on first download).</summary>
        public static string Dir
        {
            get;
        } = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".overfit", "models");

        public static void Ensure() => Directory.CreateDirectory(Dir);

        /// <summary>The cached <c>*.gguf</c> files, largest last.</summary>
        public static IReadOnlyList<FileInfo> List()
        {
            if (!Directory.Exists(Dir))
            {
                return [];
            }
            var files = new List<FileInfo>();
            foreach (var path in Directory.GetFiles(Dir, "*.gguf"))
            {
                files.Add(new FileInfo(path));
            }
            files.Sort((a, b) => string.CompareOrdinal(a.Name, b.Name));
            return files;
        }

        /// <summary>
        /// Resolves <paramref name="model"/> to a GGUF path: a direct existing file, a name in the cache
        /// (with or without the <c>.gguf</c> extension), or a case-insensitive prefix/substring match.
        /// Returns null if nothing matches.
        /// </summary>
        public static string? Resolve(string model)
        {
            if (File.Exists(model))
            {
                return model;
            }

            var direct = Path.Combine(Dir, model);
            if (File.Exists(direct))
            {
                return direct;
            }

            var withExt = model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) ? model : model + ".gguf";
            var directExt = Path.Combine(Dir, withExt);
            if (File.Exists(directExt))
            {
                return directExt;
            }

            foreach (var file in List())
            {
                var name = Path.GetFileNameWithoutExtension(file.Name);
                if (name.Equals(model, StringComparison.OrdinalIgnoreCase) ||
                    name.Contains(model, StringComparison.OrdinalIgnoreCase))
                {
                    return file.FullName;
                }
            }
            return null;
        }
    }
}
