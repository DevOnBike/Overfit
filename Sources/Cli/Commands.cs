// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.Server;

namespace DevOnBike.Overfit.Cli
{
    /// <summary>Command implementations for the <c>overfit</c> CLI.</summary>
    internal static class Commands
    {
        public static int Pull(string spec, string? file)
        {
            // A direct http(s) URL → download straight from it (internal artifact repo / approved mirror when
            // HuggingFace is blocked). Bypasses HF repo resolution entirely.
            if (Uri.TryCreate(spec, UriKind.Absolute, out var uri) && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps))
            {
                return PullFromUrl(uri);
            }

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
                var expectedSha = HfDownloader.GetExpectedSha256Async(repo, chosen).GetAwaiter().GetResult();
                HfDownloader.DownloadAsync(repo, chosen, dest, expectedSha).GetAwaiter().GetResult();
                Console.WriteLine($"Done. Chat with it:  overfit chat {Path.GetFileNameWithoutExtension(dest)}");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"pull failed: {ex.Message}");
                return 1;
            }
        }

        private static int PullFromUrl(Uri uri)
        {
            var fileName = Path.GetFileName(uri.AbsolutePath);
            if (string.IsNullOrEmpty(fileName))
            {
                Console.Error.WriteLine("Could not infer a filename from the URL — it must end with the .gguf file name.");
                return 1;
            }
            if (!fileName.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            {
                Console.Error.WriteLine($"Note: '{fileName}' does not end with .gguf — downloading anyway.");
            }

            try
            {
                ModelCache.Ensure();
                var dest = Path.Combine(ModelCache.Dir, fileName);
                if (File.Exists(dest))
                {
                    Console.WriteLine($"Already downloaded: {fileName}");
                    return 0;
                }

                Console.WriteLine($"Downloading {uri}");
                // Optional integrity check: a sibling {url}.sha256 if the server publishes one.
                var expectedSha = HfDownloader.GetSiblingSha256Async(uri.AbsoluteUri).GetAwaiter().GetResult();
                HfDownloader.DownloadUrlAsync(uri.AbsoluteUri, dest, expectedSha).GetAwaiter().GetResult();
                Console.WriteLine($"Done. Chat with it:  overfit chat {Path.GetFileNameWithoutExtension(dest)}");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"pull failed: {ex.Message}");
                return 1;
            }
        }

        private const string DefaultSystemPrompt = "You are a concise, helpful assistant running locally in pure .NET.";

        public static int Serve(string model, string host, int port)
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
                return 1;
            }

            client.AddSystem(DefaultSystemPrompt);
            var modelName = Path.GetFileNameWithoutExtension(path);

            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;   // don't kill the process; unwind the serve loop cleanly.
                cts.Cancel();
            };

            try
            {
                OverfitOpenAiServer.Serve(
                    client,
                    modelName,
                    host,
                    port,
                    DefaultSystemPrompt,
                    onListening: baseUrl =>
                    {
                        Console.WriteLine();
                        Console.WriteLine($"OpenAI-compatible server listening on {baseUrl}");
                        Console.WriteLine($"  model id:  {modelName}");
                        Console.WriteLine("  endpoints: GET /v1/models | POST /v1/chat/completions (stream + non-stream) | GET /health");
                        Console.WriteLine();
                        Console.WriteLine($"  curl {baseUrl}/v1/chat/completions -H \"Content-Type: application/json\" \\");
                        Console.WriteLine($"       -d '{{\"model\":\"{modelName}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'");
                        Console.WriteLine();
                        Console.WriteLine("Press Ctrl+C to stop.");
                    },
                    cancellationToken: cts.Token);
            }
            catch (HttpListenerException ex)
            {
                Console.Error.WriteLine($"Could not bind http://{host}:{port}/ : {ex.Message}");
                Console.Error.WriteLine("The port may be in use, or binding to a non-local host needs elevation / a URL ACL on Windows.");
                return 1;
            }
            finally
            {
                client.Dispose();
            }

            Console.WriteLine("Server stopped.");
            return 0;
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
