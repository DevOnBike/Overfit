// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Net;
using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Tts;
using DevOnBike.Overfit.Audio.Tts.Orpheus;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Embeddings;
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

            // A sentence-embedder alias (minilm / bge / e5) → download the small BERT directory, not a GGUF.
            var embedderRepo = EmbedderAliases.Resolve(spec);
            if (embedderRepo is not null)
            {
                return PullEmbedder(embedderRepo);
            }

            var resolved = ModelAliases.Resolve(spec);
            if (resolved is null)
            {
                Console.Error.WriteLine($"Unknown model '{spec}'. Pass a HuggingFace GGUF repo (owner/repo), a direct https URL, or an alias:");
                Console.Error.WriteLine("  chat:      " + string.Join(", ", ModelAliases.Known));
                Console.Error.WriteLine("  embedders: " + string.Join(", ", EmbedderAliases.Known));
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

        private static int PullEmbedder(string repo)
        {
            var dirName = repo[(repo.LastIndexOf('/') + 1)..];   // canonical folder name, e.g. all-MiniLM-L6-v2
            try
            {
                ModelCache.Ensure();
                var dir = Path.Combine(ModelCache.Dir, dirName);
                Directory.CreateDirectory(dir);
                Console.WriteLine($"Resolving embedder {repo} -> {dir}");

                foreach (var file in EmbedderAliases.Files)
                {
                    var dest = Path.Combine(dir, file);
                    if (File.Exists(dest))
                    {
                        Console.WriteLine($"  {file} already present");
                        continue;
                    }

                    Console.WriteLine($"Downloading {file} ...");
                    var sha = HfDownloader.GetExpectedSha256Async(repo, file).GetAwaiter().GetResult();
                    HfDownloader.DownloadAsync(repo, file, dest, sha).GetAwaiter().GetResult();
                }

                Console.WriteLine($"Done. Serve embeddings:  overfit serve <model> --embed-model {dirName}");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"pull failed: {ex.Message}");
                return 1;
            }
        }

        private const string DefaultSystemPrompt = "You are a concise, helpful assistant running locally in pure .NET.";

        public static int Serve(string model, string host, int port, string? embedModel)
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

            // Optional in-process sentence embedder → serves /v1/embeddings (pure .NET, no data egress).
            SentenceEmbedder? embedder = null;
            if (!string.IsNullOrWhiteSpace(embedModel))
            {
                var embedDir = ResolveEmbedderDir(embedModel);
                if (embedDir is null)
                {
                    Console.Error.WriteLine($"Embedding model '{embedModel}' not found.");
                    Console.Error.WriteLine($"Pull it first:  overfit pull minilm   (or pass a directory path with config.json + vocab.txt + model.safetensors)");
                    client.Dispose();
                    return 1;
                }

                try
                {
                    embedder = LoadEmbedder(embedDir);
                    Console.WriteLine($"Embeddings: {Path.GetFileName(embedDir.TrimEnd('\\', '/'))} ({embedder.Dimension}-dim) -> /v1/embeddings");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Failed to load embedding model '{embedDir}': {ex.Message}");
                    client.Dispose();
                    return 1;
                }
            }

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
                    embedder,
                    onListening: baseUrl =>
                    {
                        var embedEp = embedder is null ? string.Empty : " | POST /v1/embeddings";
                        Console.WriteLine();
                        Console.WriteLine($"OpenAI-compatible server listening on {baseUrl}");
                        Console.WriteLine($"  model id:  {modelName}");
                        Console.WriteLine($"  endpoints: GET /v1/models | POST /v1/chat/completions (stream + non-stream){embedEp} | GET /health");
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
                embedder?.Dispose();
                client.Dispose();
            }

            Console.WriteLine("Server stopped.");
            return 0;
        }

        /// <summary>Resolves a <c>--embed-model</c> argument to an on-disk directory: an explicit path, a known
        /// alias (minilm / bge / e5) pulled into the cache, or a cached sub-directory name. Null if not found.</summary>
        private static string? ResolveEmbedderDir(string spec)
        {
            if (Directory.Exists(spec))
            {
                return spec;
            }

            var repo = EmbedderAliases.Resolve(spec);
            if (repo is not null)
            {
                var aliasDir = Path.Combine(ModelCache.Dir, repo[(repo.LastIndexOf('/') + 1)..]);
                if (Directory.Exists(aliasDir))
                {
                    return aliasDir;
                }
            }

            var cached = Path.Combine(ModelCache.Dir, spec);
            return Directory.Exists(cached) ? cached : null;
        }

        /// <summary>Loads a sentence embedder from a HuggingFace BERT directory, picking the family convention
        /// (pooling) from the folder name: <c>bge</c> → CLS, <c>e5</c> → Mean+prefixes, else MiniLM (Mean).</summary>
        private static SentenceEmbedder LoadEmbedder(string dir)
        {
            var name = Path.GetFileName(dir.TrimEnd('\\', '/')).ToLowerInvariant();
            if (name.Contains("bge"))
            {
                return SentenceEmbedder.ForBgeEnV15(dir);
            }
            if (name.Contains("e5"))
            {
                return SentenceEmbedder.ForE5(dir);
            }
            return SentenceEmbedder.ForMiniLm(dir);
        }

        private static string VoicesDir() => Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".overfit", "voices");

        public static int Tts(string text, string voice, string outPath, string language, string? model, string? snacDir)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                Console.Error.WriteLine("--text is required.");
                return 1;
            }

            try
            {
                var orpheus = ResolveOrpheusModel(model);
                var snac = ResolveSnacDir(snacDir);
                if (orpheus is not null && snac is not null)
                {
                    return TtsOrpheus(text, voice, outPath, orpheus, snac);
                }

                return TtsPlaceholder(text, voice, language, outPath, orpheus, snac);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"tts failed: {ex.Message}");
                return 1;
            }
        }

        // Real neural TTS: Orpheus LM → SNAC decoder → watermarked WAV, pure managed CPU.
        private static int TtsOrpheus(string text, string voice, string outPath, string orpheusGguf, string snacDir)
        {
            var chosen = OrpheusPrompt.IsKnownVoice(voice) ? voice : OrpheusPrompt.DefaultVoice;
            if (!OrpheusPrompt.IsKnownVoice(voice))
            {
                Console.WriteLine($"Voice '{voice}' is not an Orpheus preset; using '{chosen}'. "
                    + $"Presets: {string.Join(", ", OrpheusPrompt.AvailableVoices)}.");
            }

            Console.WriteLine($"Synthesizing with Orpheus + SNAC on CPU (voice '{chosen}')…");
            using var engine = OrpheusVoiceEngine.Load(orpheusGguf, snacDir);
            var audio = engine.Synthesize(text, chosen);

            var marker = SyntheticSpeechMetadata.ForNow(chosen);
            using (var sink = new WavAudioSink(outPath, engine.SampleRate, WavSampleFormat.Pcm16, marker))
            {
                sink.Write(audio);
            }

            var seconds = audio.Length / (double)engine.SampleRate;
            Console.WriteLine($"Wrote {outPath}  ({seconds:F2}s, voice '{chosen}', {engine.SampleRate} Hz, synthetic — watermarked).");
            return 0;
        }

        // Fallback tone engine when the neural models are not installed — keeps `overfit tts` usable and explains how
        // to enable real speech.
        private static int TtsPlaceholder(string text, string voice, string language, string outPath, string? orpheus, string? snac)
        {
            var voicesDir = VoicesDir();
            var profile = VoiceProfileStore.Exists(voice, voicesDir)
                ? VoiceProfileStore.Load(voice, voicesDir)
                : VoiceProfile.Preset(voice, language);

            var engine = new PlaceholderTtsEngine(24000);
            var marker = SyntheticSpeechMetadata.ForNow(profile.Id);
            using (var sink = new WavAudioSink(outPath, engine.SampleRate, WavSampleFormat.Pcm16, marker))
            {
                engine.Synthesize(text, profile, sink, TtsOptions.Default);
            }

            Console.WriteLine($"Wrote {outPath}  (voice '{profile.Id}', {engine.SampleRate} Hz, placeholder tone — watermarked).");
            Console.WriteLine();
            Console.WriteLine("This is the placeholder tone engine — real neural speech needs two models:");
            Console.WriteLine(orpheus is null
                ? "  • Orpheus GGUF: overfit pull isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF   (then pass --model <file.gguf>)"
                : $"  • Orpheus GGUF: found ({orpheus})");
            Console.WriteLine(snac is null
                ? "  • SNAC weights: python Scripts/convert_snac.py --out %USERPROFILE%\\.overfit\\snac   (or pass --snac <dir>)"
                : $"  • SNAC weights: found ({snac})");
            return 0;
        }

        // Orpheus GGUF: explicit --model (cache name or path), else $OVERFIT_ORPHEUS_DIR, else any orpheus*.gguf in the cache.
        private static string? ResolveOrpheusModel(string? model)
        {
            if (!string.IsNullOrWhiteSpace(model))
            {
                return ModelCache.Resolve(model) ?? (File.Exists(model) ? model : null);
            }

            var envDir = Environment.GetEnvironmentVariable("OVERFIT_ORPHEUS_DIR");
            if (!string.IsNullOrWhiteSpace(envDir) && Directory.Exists(envDir))
            {
                var hit = FindOrpheusGguf(envDir);
                if (hit is not null)
                {
                    return hit;
                }
            }

            return Directory.Exists(ModelCache.Dir) ? FindOrpheusGguf(ModelCache.Dir) : null;
        }

        private static string? FindOrpheusGguf(string dir)
        {
            foreach (var path in Directory.GetFiles(dir, "*.gguf"))
            {
                if (Path.GetFileName(path).Contains("orpheus", StringComparison.OrdinalIgnoreCase))
                {
                    return path;
                }
            }
            return null;
        }

        // SNAC dir: explicit --snac, else $OVERFIT_SNAC_DIR, else ~/.overfit/snac. Must contain the converted weights.
        private static string? ResolveSnacDir(string? snacDir)
        {
            var candidates = new List<string?>
            {
                snacDir,
                Environment.GetEnvironmentVariable("OVERFIT_SNAC_DIR"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".overfit", "snac"),
            };
            foreach (var c in candidates)
            {
                if (!string.IsNullOrWhiteSpace(c) && File.Exists(Path.Combine(c, "snac_24khz.safetensors")))
                {
                    return c;
                }
            }
            return null;
        }

        public static int TtsEval(string referencePath, string candidatePath)
        {
            try
            {
                if (!File.Exists(referencePath))
                {
                    Console.Error.WriteLine($"Reference audio not found: {referencePath}");
                    return 1;
                }
                if (!File.Exists(candidatePath))
                {
                    Console.Error.WriteLine($"Candidate audio not found: {candidatePath}");
                    return 1;
                }

                var reference = AudioFile.ReadMono(referencePath, out var refRate);
                var candidate = AudioFile.ReadMono(candidatePath, out var candRate);

                var report = AudioSimilarity.Compare(reference, refRate, candidate, candRate);

                Console.WriteLine($"reference : {Path.GetFileName(referencePath)}  ({refRate} Hz, {reference.Length} samples)");
                Console.WriteLine($"candidate : {Path.GetFileName(candidatePath)}  ({candRate} Hz, {candidate.Length} samples)");
                Console.WriteLine();
                var snr = double.IsPositiveInfinity(report.SignalToNoiseRatioDb)
                    ? "inf (bit-identical)"
                    : report.SignalToNoiseRatioDb.ToString("0.0", CultureInfo.InvariantCulture) + " dB";
                Console.WriteLine($"  SNR (waveform)        {snr}");
                Console.WriteLine($"  correlation           {report.Correlation:0.000}");
                Console.WriteLine($"  RMSE                  {report.RootMeanSquareError:0.0000}");
                Console.WriteLine($"  mel distance          {report.MelSpectralDistance:0.0000}");
                Console.WriteLine($"  mel distance (DTW)    {report.MelDistanceDtw:0.0000}   <- timing-robust; lower = closer to ideal");
                Console.WriteLine();
                Console.WriteLine("Waveform metrics (SNR / correlation) assume the two are the same deterministic decode");
                Console.WriteLine("(aligned sample-for-sample). For generated speech vs. a reference clip, the DTW mel");
                Console.WriteLine("distance is the metric that matters.");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"tts eval failed: {ex.Message}");
                return 1;
            }
        }

        public static int VoiceEnroll(string id, string sample, string language, bool consent)
        {
            if (!consent)
            {
                Console.Error.WriteLine("Voice enrollment requires consent. Re-run with --consent to confirm you OWN");
                Console.Error.WriteLine("this voice or have explicit permission to use it. Cloning a voice without");
                Console.Error.WriteLine("consent may be illegal in your jurisdiction.");
                return 1;
            }

            try
            {
                if (!File.Exists(sample))
                {
                    Console.Error.WriteLine($"Sample audio not found: {sample}");
                    return 1;
                }

                // Validate the clip decodes; we don't compute a speaker embedding yet (gated Phase 2).
                var pcm = AudioFile.ReadMono(sample, out var rate);
                var seconds = pcm.Length / (double)rate;

                var voicesDir = VoicesDir();
                var profile = new VoiceProfile(id, language, speakerEmbedding: null, referenceAudioPath: Path.GetFullPath(sample));
                VoiceProfileStore.Save(profile, voicesDir);

                Console.WriteLine($"Enrolled voice '{id}' ({language}) from {Path.GetFileName(sample)} ({seconds:F1}s) -> {voicesDir}");
                Console.WriteLine("Note: a speaker embedding is NOT computed yet — voice cloning is a gated Phase 2");
                Console.WriteLine("(docs/tts-poc-plan.md). The reference clip is recorded for then.");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"enroll failed: {ex.Message}");
                return 1;
            }
        }

        public static int VoiceList()
        {
            var dir = VoicesDir();
            var ids = VoiceProfileStore.List(dir);
            if (ids.Count == 0)
            {
                Console.WriteLine($"No enrolled voices in {dir}.");
                Console.WriteLine("Enroll one:  overfit voice enroll <id> --sample your.wav --consent");
                return 0;
            }

            Console.WriteLine($"Voices in {dir}:");
            Console.WriteLine();
            foreach (var voiceId in ids)
            {
                var profile = VoiceProfileStore.Load(voiceId, dir);
                Console.WriteLine($"  {profile.Id,-24} {profile.Language,-4} {(profile.IsCloned ? "cloned" : "preset")}");
            }
            return 0;
        }

        public static int List()
        {
            var models = ModelCache.List();
            var embedders = ListEmbedderDirs();

            if (models.Count == 0 && embedders.Count == 0)
            {
                Console.WriteLine($"No models in {ModelCache.Dir}.");
                Console.WriteLine("Download one, e.g.:  overfit pull qwen2.5-3b   (chat)   |   overfit pull minilm   (embeddings)");
                return 0;
            }

            Console.WriteLine($"Models in {ModelCache.Dir}:");
            Console.WriteLine();
            foreach (var file in models)
            {
                var name = Path.GetFileNameWithoutExtension(file.Name);
                var gb = file.Length / (1024.0 * 1024 * 1024);
                Console.WriteLine($"  {name,-44} {gb,6:F2} GB   chat");
            }
            foreach (var dir in embedders)
            {
                Console.WriteLine($"  {dir,-44} {string.Empty,6}      embedder");
            }
            return 0;
        }

        /// <summary>Cached sentence-embedder directories (a sub-folder with a <c>model.safetensors</c>).</summary>
        private static IReadOnlyList<string> ListEmbedderDirs()
        {
            var result = new List<string>();
            if (!Directory.Exists(ModelCache.Dir))
            {
                return result;
            }

            foreach (var dir in Directory.GetDirectories(ModelCache.Dir))
            {
                if (File.Exists(Path.Combine(dir, "model.safetensors")))
                {
                    result.Add(Path.GetFileName(dir));
                }
            }
            return result;
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
