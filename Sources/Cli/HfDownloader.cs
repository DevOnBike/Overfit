// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json;
using DevOnBike.Overfit.Diagnostics;

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

        /// <summary>Base HuggingFace endpoint. Override with the <c>HF_ENDPOINT</c> env var (same convention as
        /// the official <c>huggingface_hub</c>) to point at a mirror or an internal proxy when
        /// <c>huggingface.co</c> is blocked — e.g. <c>HF_ENDPOINT=https://hf-mirror.com</c>.</summary>
        private static readonly string Endpoint = ResolveEndpoint();

        private static string ResolveEndpoint()
        {
            var endpoint = Environment.GetEnvironmentVariable("HF_ENDPOINT");
            return string.IsNullOrWhiteSpace(endpoint) ? "https://huggingface.co" : endpoint.TrimEnd('/');
        }

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
            using var response = await Http.GetAsync($"{Endpoint}/api/models/{repo}");

            if (response.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden)
            {
                throw new InvalidOperationException($"Repo '{repo}' is gated or private. Accept its terms on huggingface.co and set HF_TOKEN.");
            }

            if (response.StatusCode == HttpStatusCode.NotFound)
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
                    if (g.Equals(explicitFile, StringComparison.OrdinalIgnoreCase) || g.Contains(explicitFile, StringComparison.OrdinalIgnoreCase))
                    {
                        return g;
                    }
                }

                throw new InvalidOperationException($"No .gguf matching '{explicitFile}' in '{repo}'. Available:\n  " + string.Join("\n  ", ggufs));
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

        /// <summary>
        /// Fetches the expected SHA-256 (the LFS object id) for <paramref name="file"/> from the repo's git
        /// tree (<c>/api/models/{repo}/tree/main?recursive=true</c>). Returns null when the file isn't
        /// LFS-backed or the metadata can't be retrieved — the caller then skips verification rather than
        /// failing the download (a transient API hiccup shouldn't block a pull).
        /// </summary>
        public static async Task<string?> GetExpectedSha256Async(string repo, string file)
        {
            try
            {
                using var response = await Http.GetAsync($"{Endpoint}/api/models/{repo}/tree/main?recursive=true");
                if (!response.IsSuccessStatusCode)
                {
                    return null;
                }

                var json = await response.Content.ReadAsStringAsync();
                using var doc = JsonDocument.Parse(json);
                if (doc.RootElement.ValueKind != JsonValueKind.Array)
                {
                    return null;
                }

                foreach (var entry in doc.RootElement.EnumerateArray())
                {
                    if (entry.TryGetProperty("path", out var p) && string.Equals(p.GetString(), file, StringComparison.Ordinal)
                        && entry.TryGetProperty("lfs", out var lfs) && lfs.ValueKind == JsonValueKind.Object
                        && lfs.TryGetProperty("oid", out var oid) && oid.ValueKind == JsonValueKind.String)
                    {
                        return oid.GetString();
                    }
                }

                return null;
            }
            catch
            {
                return null;   // network / parse issue → skip verification, don't abort the pull.
            }
        }

        /// <summary>
        /// For a direct-URL download, tries the conventional sibling checksum file <c>{url}.sha256</c> (its first
        /// whitespace-delimited token is the hex digest, per <c>sha256sum</c> output). Returns null when absent or
        /// malformed — verification is then skipped rather than failing the download.
        /// </summary>
        public static async Task<string?> GetSiblingSha256Async(string url)
        {
            try
            {
                using var response = await Http.GetAsync($"{url}.sha256");
                if (!response.IsSuccessStatusCode)
                {
                    return null;
                }

                var text = await response.Content.ReadAsStringAsync();
                var parts = text.Split([' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length > 0 && parts[0].Length == 64 && IsHex(parts[0]))
                {
                    return parts[0].ToLowerInvariant();
                }

                return null;
            }
            catch
            {
                return null;
            }
        }

        private static bool IsHex(string s)
        {
            foreach (var c in s)
            {
                if (!Uri.IsHexDigit(c))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>Streams <paramref name="file"/> from <paramref name="repo"/> to <paramref name="destPath"/>
        /// (via a <c>.part</c> temp), printing a single-line progress indicator. RESUMES an interrupted prior
        /// download: a leftover <c>.part</c> is continued from its current length via an HTTP <c>Range</c>
        /// request (server must answer 206; a 200 means Range was ignored, so it restarts). Computes the
        /// content's SHA-256 while streaming — including the bytes already on disk on resume — and, when
        /// <paramref name="expectedSha256"/> is supplied, verifies it BEFORE the temp is promoted to the final
        /// path, so a corrupt or mis-resumed download is discarded and never lands under its real name.</summary>
        public static Task DownloadAsync(string repo, string file, string destPath, string? expectedSha256)
            => StreamWithResumeAsync($"{Endpoint}/{repo}/resolve/main/{file}", file, destPath, expectedSha256);

        /// <summary>Downloads a GGUF straight from an absolute <paramref name="url"/> (e.g. an internal artifact
        /// repository or an approved mirror when HuggingFace is unreachable) into <paramref name="destPath"/>,
        /// with the same resume + SHA-256 verification as the repo path. The display name comes from the URL.</summary>
        public static Task DownloadUrlAsync(string url, string destPath, string? expectedSha256)
            => StreamWithResumeAsync(url, Path.GetFileName(new Uri(url).AbsolutePath), destPath, expectedSha256);

        private static async Task StreamWithResumeAsync(string url, string file, string destPath, string? expectedSha256)
        {
            var tmp = destPath + ".part";
            var existing = File.Exists(tmp) ? new FileInfo(tmp).Length : 0L;

            using var hasher = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

            using var request = new HttpRequestMessage(HttpMethod.Get, url);
            if (existing > 0)
            {
                request.Headers.Range = new RangeHeaderValue(existing, null);
            }

            using var response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);

            if (existing > 0 && response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
            {
                // The .part already holds the whole file (a prior run finished the bytes but was interrupted
                // before verification) — hash what's on disk and fall straight through to verify + promote.
                await HashExistingAsync(tmp, hasher);
                Console.WriteLine($"  {file}  already downloaded ({existing / (1024.0 * 1024):F1} MB) — verifying");
            }
            else
            {
                response.EnsureSuccessStatusCode();

                var resuming = existing > 0 && response.StatusCode == HttpStatusCode.PartialContent;
                if (existing > 0 && !resuming)
                {
                    existing = 0;   // server ignored the Range (200 OK) → start over from scratch.
                }

                long total;
                if (response.Content.Headers.ContentRange?.Length is long full)
                {
                    total = full;
                }
                else if (resuming)
                {
                    total = existing + (response.Content.Headers.ContentLength ?? 0);
                }
                else
                {
                    total = response.Content.Headers.ContentLength ?? -1;
                }

                if (resuming)
                {
                    // Fold the already-downloaded prefix into the digest so the final hash covers the whole file.
                    await HashExistingAsync(tmp, hasher);
                    Console.WriteLine($"  resuming {file} from {existing / (1024.0 * 1024):F1} MB ...");
                }

                await using (var source = await response.Content.ReadAsStreamAsync())
                await using (var dest = new FileStream(tmp, existing > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    var buffer = new byte[1 << 20];
                    var read = existing;
                    var lastReport = existing;
                    var stopwatch = ValueStopwatch.StartNew();
                    
                    int n;
                    
                    while ((n = await source.ReadAsync(buffer)) > 0)
                    {
                        await dest.WriteAsync(buffer.AsMemory(0, n));
                        hasher.AppendData(buffer, 0, n);
                        read += n;
                        
                        if (read - lastReport >= (4L << 20))
                        {
                            Report(file, read, total, read - existing, stopwatch.GetElapsedTime().TotalSeconds);
                            
                            lastReport = read;
                        }
                    }
                    
                    Report(file, read, total, read - existing, stopwatch.GetElapsedTime().TotalSeconds);
                }
                Console.WriteLine();
            }

            var actual = Convert.ToHexString(hasher.GetHashAndReset()).ToLowerInvariant();
            if (expectedSha256 is not null)
            {
                if (!string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
                {
                    File.Delete(tmp);
                    throw new InvalidOperationException(
                        $"SHA-256 mismatch for '{file}' — the download is corrupt and was discarded.\n" +
                        $"  expected: {expectedSha256}\n  actual:   {actual}");
                }

                Console.WriteLine($"  sha256 verified  {actual[..16]}...");
            }
            else
            {
                Console.WriteLine($"  sha256 {actual[..16]}...  (HF metadata unavailable — not verified)");
            }

            File.Move(tmp, destPath, overwrite: true);
        }

        /// <summary>Feeds the bytes already present in <paramref name="path"/> into <paramref name="hasher"/>
        /// (used on resume so the running digest covers the whole file, not just the newly fetched tail).</summary>
        private static async Task HashExistingAsync(string path, IncrementalHash hasher)
        {
            await using var stream = File.OpenRead(path);
            var buffer = new byte[1 << 20];
            int n;
            while ((n = await stream.ReadAsync(buffer)) > 0)
            {
                hasher.AppendData(buffer, 0, n);
            }
        }

        // sessionBytes = bytes fetched in THIS run (so the rate is honest on resume, where `read` starts mid-file).
        private static void Report(string file, long read, long total, long sessionBytes, double seconds)
        {
            var mb = read / (1024.0 * 1024);
            var speed = seconds > 0 ? (sessionBytes / (1024.0 * 1024)) / seconds : 0;

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
