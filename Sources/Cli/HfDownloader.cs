// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Text.Json;

namespace DevOnBike.Overfit.Cli
{
    /// <summary>
    /// Downloads a GGUF model file from a HuggingFace repo: lists the repo's <c>.gguf</c> siblings via the
    /// HF API, picks one (an explicit name, a quant pattern, or the first), and streams it to the local
    /// store with a progress line. A gated repo (e.g. Bielik) needs an <c>HF_TOKEN</c> env var.
    /// </summary>
    internal static class HfDownloader
    {
        private static readonly HttpClient Http = CreateClient();

        private static HttpClient CreateClient()
        {
            var client = new HttpClient(new HttpClientHandler { AllowAutoRedirect = true })
            {
                Timeout = Timeout.InfiniteTimeSpan,   // model files are large
            };
            client.DefaultRequestHeaders.UserAgent.ParseAdd("overfit-cli/1.0");
            var token = Environment.GetEnvironmentVariable("HF_TOKEN");
            if (!string.IsNullOrWhiteSpace(token))
            {
                client.DefaultRequestHeaders.Authorization = new("Bearer", token);
            }
            return client;
        }

        /// <summary>Picks the GGUF file to download from <paramref name="repo"/>: an explicit name, else the
        /// one matching <paramref name="pattern"/> (a quant like "q4_k_m"), else the first. Throws with the
        /// available files if nothing matches.</summary>
        public static async Task<string> ResolveFileAsync(string repo, string? pattern, string? explicitFile)
        {
            using var response = await Http.GetAsync($"https://huggingface.co/api/models/{repo}");
            if (response.StatusCode is System.Net.HttpStatusCode.Unauthorized or System.Net.HttpStatusCode.Forbidden)
            {
                throw new InvalidOperationException(
                    $"Repo '{repo}' is gated or private. Accept its terms on huggingface.co and set HF_TOKEN.");
            }
            if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                throw new InvalidOperationException($"HuggingFace repo '{repo}' was not found.");
            }
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(json);

            var ggufs = new List<string>();
            if (doc.RootElement.TryGetProperty("siblings", out var siblings) && siblings.ValueKind == JsonValueKind.Array)
            {
                foreach (var sibling in siblings.EnumerateArray())
                {
                    if (sibling.TryGetProperty("rfilename", out var rf) && rf.GetString() is { } name &&
                        name.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                    {
                        ggufs.Add(name);
                    }
                }
            }
            if (ggufs.Count == 0)
            {
                throw new InvalidOperationException($"Repo '{repo}' has no .gguf files.");
            }

            if (explicitFile is not null)
            {
                foreach (var g in ggufs)
                {
                    if (g.Equals(explicitFile, StringComparison.OrdinalIgnoreCase) ||
                        g.Contains(explicitFile, StringComparison.OrdinalIgnoreCase))
                    {
                        return g;
                    }
                }
                throw new InvalidOperationException(
                    $"No .gguf matching '{explicitFile}' in '{repo}'. Available:\n  " + string.Join("\n  ", ggufs));
            }

            if (pattern is not null)
            {
                foreach (var g in ggufs)
                {
                    if (g.Contains(pattern, StringComparison.OrdinalIgnoreCase))
                    {
                        return g;
                    }
                }
            }
            return ggufs[0];
        }

        /// <summary>Streams <paramref name="file"/> from <paramref name="repo"/> to <paramref name="destPath"/>
        /// (via a <c>.part</c> temp), printing a single-line progress indicator.</summary>
        public static async Task DownloadAsync(string repo, string file, string destPath)
        {
            var url = $"https://huggingface.co/{repo}/resolve/main/{file}";
            using var response = await Http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            var total = response.Content.Headers.ContentLength ?? -1;
            var tmp = destPath + ".part";

            await using (var source = await response.Content.ReadAsStreamAsync())
            await using (var dest = File.Create(tmp))
            {
                var buffer = new byte[1 << 20];
                long read = 0;
                long lastReport = 0;
                var stopwatch = Stopwatch.StartNew();
                int n;
                while ((n = await source.ReadAsync(buffer)) > 0)
                {
                    await dest.WriteAsync(buffer.AsMemory(0, n));
                    read += n;
                    if (read - lastReport >= (4L << 20))
                    {
                        Report(file, read, total, stopwatch.Elapsed.TotalSeconds);
                        lastReport = read;
                    }
                }
                Report(file, read, total, stopwatch.Elapsed.TotalSeconds);
            }
            Console.WriteLine();
            File.Move(tmp, destPath, overwrite: true);
        }

        private static void Report(string file, long read, long total, double seconds)
        {
            var mb = read / (1024.0 * 1024);
            var speed = seconds > 0 ? mb / seconds : 0;
            if (total > 0)
            {
                var pct = read * 100.0 / total;
                var totalMb = total / (1024.0 * 1024);
                Console.Write($"\r  {file}  {pct,5:F1}%  ({mb,8:F1} / {totalMb:F1} MB)  {speed,6:F1} MB/s    ");
            }
            else
            {
                Console.Write($"\r  {file}  {mb,8:F1} MB  {speed,6:F1} MB/s    ");
            }
        }
    }
}
