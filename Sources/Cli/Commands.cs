// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels;

namespace DevOnBike.Overfit.Cli
{
    /// <summary>Command implementations for the <c>overfit</c> CLI.</summary>
    internal static class Commands
    {
        public static int Pull(string spec, string? file)
        {
            var resolved = ModelAliases.Resolve(spec);
            if (resolved is null)
            {
                Console.Error.WriteLine($"Unknown model '{spec}'. Pass a HuggingFace GGUF repo (owner/repo) or an alias:");
                Console.Error.WriteLine("  " + string.Join(", ", ModelAliases.Known));
                return 1;
            }

            var (repo, pattern) = resolved.Value;
            try
            {
                ModelCache.Ensure();
                Console.WriteLine($"Resolving {repo} ...");
                var chosen = HfDownloader.ResolveFileAsync(repo, pattern, file).GetAwaiter().GetResult();
                var dest = Path.Combine(ModelCache.Dir, Path.GetFileName(chosen));

                if (File.Exists(dest))
                {
                    Console.WriteLine($"Already downloaded: {Path.GetFileName(dest)}");
                    return 0;
                }

                Console.WriteLine($"Downloading {chosen} into {ModelCache.Dir}");
                HfDownloader.DownloadAsync(repo, chosen, dest).GetAwaiter().GetResult();
                Console.WriteLine($"Done. Chat with it:  overfit chat {Path.GetFileNameWithoutExtension(dest)}");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"pull failed: {ex.Message}");
                return 1;
            }
        }

        public static int Serve(string model)
        {
            Console.Error.WriteLine($"serve '{model}': not implemented in this build (OpenAI server coming next).");
            return 1;
        }

        public static int List()
        {
            var models = ModelCache.List();
            if (models.Count == 0)
            {
                Console.WriteLine($"No models in {ModelCache.Dir}.");
                Console.WriteLine("Download one, e.g.:  overfit pull qwen2.5-3b");
                return 0;
            }

            Console.WriteLine($"Models in {ModelCache.Dir}:");
            Console.WriteLine();
            foreach (var file in models)
            {
                var name = Path.GetFileNameWithoutExtension(file.Name);
                var gb = file.Length / (1024.0 * 1024 * 1024);
                Console.WriteLine($"  {name,-44} {gb,6:F2} GB");
            }
            return 0;
        }

        public static int Chat(string model)
        {
            var path = ModelCache.Resolve(model);
            if (path is null)
            {
                Console.Error.WriteLine($"Model '{model}' not found in {ModelCache.Dir}.");
                Console.Error.WriteLine($"Download it first:  overfit pull {model}   (or pass a .gguf path directly)");
                return 1;
            }

            Console.WriteLine($"Loading {Path.GetFileName(path)} ...");
            OverfitClient client;
            try
            {
                client = OverfitClient.LoadGguf(path, mmap: true);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed to load '{Path.GetFileName(path)}': {ex.Message}");
                Console.Error.WriteLine("The model's quantization may not be supported yet (e.g. Q5_0/Q4_0 used by some");
                Console.Error.WriteLine("small-model GGUFs whose hidden size is not a multiple of 256). Try a Q8_0 build,");
                Console.Error.WriteLine("or a model whose hidden size is a multiple of 256 (e.g. qwen2.5-3b).");
                return 1;
            }

            client.AddSystem("You are a concise, helpful assistant running locally in pure .NET.");
            Console.WriteLine("Ready. Type a message; /reset clears the conversation, /exit quits.");
            Console.WriteLine();

            try
            {
                while (true)
                {
                    Console.Write("> ");
                    var line = Console.ReadLine();
                    if (line is null || line.Equals("/exit", StringComparison.OrdinalIgnoreCase))
                    {
                        break;
                    }
                    if (line.Equals("/reset", StringComparison.OrdinalIgnoreCase))
                    {
                        client.Reset();
                        client.AddSystem("You are a concise, helpful assistant running locally in pure .NET.");
                        Console.WriteLine("(conversation cleared)");
                        Console.WriteLine();
                        continue;
                    }
                    if (string.IsNullOrWhiteSpace(line))
                    {
                        continue;
                    }

                    client.Send(line, onText: Console.Write);
                    var stats = client.Chat.LastStats;
                    Console.WriteLine();
                    Console.WriteLine($"  [{stats.GeneratedTokens} tokens, {stats.TokensPerSecond:F1} tok/s]");
                    Console.WriteLine();
                }
            }
            finally
            {
                client.Dispose();
            }
            return 0;
        }
    }
}
