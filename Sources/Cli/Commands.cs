// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Net;
using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Tts;
using DevOnBike.Overfit.Audio.Tts.Orpheus;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Embeddings;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Whisper;
using DevOnBike.Overfit.Mcp;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Redaction;
using DevOnBike.Overfit.Server;
using DevOnBike.Overfit.Serving;
using DevOnBike.Overfit.Trees;

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

        public static int Serve(string model, string host, int port, string? embedModel, string? ttsModel, string? ttsSnac, int sessions = 1)
        {
            var path = ModelCache.Resolve(model);
            if (path is null)
            {
                Console.Error.WriteLine($"Model '{model}' not found in {ModelCache.Dir}.");
                Console.Error.WriteLine($"Download it first:  overfit pull {model}   (or pass a .gguf path directly)");
                return 1;
            }

            if (sessions < 1)
            {
                sessions = 1;
            }

            // One client per session — each owns its KV cache (extra RAM); the model weights are shared via
            // mmap (the OS page cache de-duplicates them), so N sessions ≈ 1× weights + N× KV.
            Console.WriteLine(sessions == 1
                ? $"Loading {Path.GetFileName(path)} ..."
                : $"Loading {Path.GetFileName(path)} into {sessions} sessions ...");
            var clients = new List<OverfitClient>(sessions);
            try
            {
                for (var i = 0; i < sessions; i++)
                {
                    var c = OverfitClient.LoadGguf(path, mmap: true);
                    c.AddSystem(DefaultSystemPrompt);
                    clients.Add(c);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed to load '{Path.GetFileName(path)}': {ex.Message}");
                foreach (var c in clients)
                {
                    c.Dispose();
                }
                return 1;
            }

            var pool = new OverfitResourcePool<OverfitClient>(clients);
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
                    pool.Dispose();
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
                    pool.Dispose();
                    return 1;
                }
            }

            // Optional in-process TTS → serves /v1/audio/speech (Orpheus LM + SNAC, pure .NET, no data egress).
            OrpheusVoiceEngine? tts = null;
            if (!string.IsNullOrWhiteSpace(ttsModel))
            {
                var orpheus = ResolveOrpheusModel(ttsModel);
                var snac = ResolveSnacDir(ttsSnac);
                if (orpheus is null || snac is null)
                {
                    Console.Error.WriteLine("TTS not started: need both an Orpheus GGUF (--tts-model) and SNAC weights (--tts-snac).");
                    Console.Error.WriteLine(orpheus is null ? "  • Orpheus GGUF not found (pull it, or pass --tts-model <file.gguf>)." : $"  • Orpheus: {orpheus}");
                    Console.Error.WriteLine(snac is null ? "  • SNAC weights not found (run Scripts/convert_snac.py, or pass --tts-snac <dir>)." : $"  • SNAC: {snac}");
                    embedder?.Dispose();
                    pool.Dispose();
                    return 1;
                }

                try
                {
                    tts = OrpheusVoiceEngine.Load(orpheus, snac);
                    Console.WriteLine($"TTS: Orpheus + SNAC -> /v1/audio/speech (voices: {string.Join(", ", OrpheusPrompt.AvailableVoices)})");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Failed to load TTS: {ex.Message}");
                    embedder?.Dispose();
                    pool.Dispose();
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
                    pool,
                    modelName,
                    host,
                    port,
                    DefaultSystemPrompt,
                    embedder,
                    tts,
                    onListening: baseUrl =>
                    {
                        var embedEp = embedder is null ? string.Empty : " | POST /v1/embeddings";
                        var ttsEp = tts is null ? string.Empty : " | POST /v1/audio/speech";
                        Console.WriteLine();
                        Console.WriteLine($"Overfit OpenAI-compatible server listening on {baseUrl}");
                        Console.WriteLine($"  model id:  {modelName}");
                        Console.WriteLine($"  endpoints: GET /v1/models | POST /v1/chat/completions (stream + non-stream){embedEp}{ttsEp} | GET /health");
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
                tts?.Dispose();
                embedder?.Dispose();
                pool.Dispose();
            }

            Console.WriteLine("Server stopped.");
            return 0;
        }

        /// <summary>
        /// `overfit doctor &lt;model&gt;` — inspect a GGUF and report what Overfit sees: architecture, quant,
        /// tokenizer, chat template, context, whether the arch is supported, recommended tuning flags, and
        /// warnings. Read-only (parses metadata + tensor headers; does not load weights) — answers the #1
        /// adoption question, "why doesn't my model work / is the tokenizer + template detected".
        /// </summary>
        public static int Doctor(string model)
        {
            var path = ModelCache.Resolve(model);
            if (path is null)
            {
                Console.Error.WriteLine($"Model '{model}' not found in {ModelCache.Dir}.");
                Console.Error.WriteLine($"Download it first:  overfit pull {model}   (or pass a .gguf path directly)");
                return 1;
            }

            GgufReader reader;
            try
            {
                reader = new GgufReader(path);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed to read '{Path.GetFileName(path)}' as GGUF: {ex.Message}");
                return 1;
            }

            using (reader)
            {
                var arch = reader.GetMeta("general.architecture", "?");
                var name = reader.GetMeta("general.name", string.Empty);
                var layers = reader.GetMeta($"{arch}.block_count", 0);
                var hidden = reader.GetMeta($"{arch}.embedding_length", 0);
                var heads = reader.GetMeta($"{arch}.attention.head_count", 0);
                var kvHeads = reader.GetMeta($"{arch}.attention.head_count_kv", heads);
                var ctx = reader.GetMeta($"{arch}.context_length", 0);
                var experts = reader.GetMeta($"{arch}.expert_count", 0);
                var tokModel = reader.GetMeta("tokenizer.ggml.model", "?");
                var hasTemplate = reader.Metadata.ContainsKey("tokenizer.chat_template");

                var (quant, quantMixed, quantizedFraction) = DescribeQuant(reader);
                var (supported, supportNote) = DescribeSupport(arch, experts);

                Console.WriteLine();
                Console.WriteLine($"overfit doctor — {path}");
                Console.WriteLine();
                Console.WriteLine($"  architecture   : {arch}{(string.IsNullOrEmpty(name) ? string.Empty : $"  ({name})")}");
                if (layers > 0)
                {
                    var gqa = kvHeads > 0 && kvHeads < heads ? $", GQA {heads}:{kvHeads}" : string.Empty;
                    var moe = experts > 0 ? $", MoE ×{experts} experts" : string.Empty;
                    Console.WriteLine($"  parameters     : {layers} layers · {hidden} hidden · {heads} heads{gqa}{moe}");
                }
                Console.WriteLine($"  quantization   : {quant}{(quantMixed ? "  (mixed)" : string.Empty)}  —  {quantizedFraction:P0} of weight tensors quantized");
                Console.WriteLine($"  tokenizer      : {DescribeTokenizer(tokModel)} (embedded in GGUF)");
                Console.WriteLine($"  chat template  : {(hasTemplate ? "found" : "MISSING — chat formatting falls back to a generic template")}");
                Console.WriteLine($"  context length : {(ctx > 0 ? ctx.ToString(CultureInfo.InvariantCulture) : "unknown")}");
                Console.WriteLine($"  supported      : {supported} — {supportNote}");

                Console.WriteLine();
                Console.WriteLine("  recommended flags:");
                if (quant.StartsWith("Q4_K", StringComparison.Ordinal))
                {
                    Console.WriteLine($"    {OverfitEnvironment.RepackGemv}=1      # ~+30% decode on Q4_K (FFN + LM-head repacked 8×8 GEMV)");
                }
                Console.WriteLine($"    {OverfitEnvironment.DecodeWorkers}=<n>   # match physical cores (16 was optimal with repack on a 16-core box)");

                var warnings = CollectWarnings(arch, hasTemplate, supported, tokModel);
                if (warnings.Count > 0)
                {
                    Console.WriteLine();
                    Console.WriteLine("  warnings:");
                    foreach (var w in warnings)
                    {
                        Console.WriteLine($"    • {w}");
                    }
                }

                Console.WriteLine();
            }

            return 0;
        }

        // ── gateway: LLM egress firewall — redact outbound PII/secrets, forward upstream (gateway holds the key). ──
        public static int Gateway(string? upstream, string keyEnv, string host, int port, string auditPath, string? configPath, string clientKeysEnv, bool insecure, bool scanResponses)
        {
            // TLS guard: HttpListener serves plaintext HTTP. Binding that to a non-loopback interface would send the
            // very PII/secrets this gateway protects across the network unencrypted. Refuse unless explicitly waived —
            // the supported pattern is a TLS-terminating proxy/sidecar in front, with the gateway bound to loopback.
            if (!IsLoopbackHost(host))
            {
                if (!insecure)
                {
                    Console.Error.WriteLine(
                        $"Refusing to bind a plaintext HTTP gateway to non-loopback host '{host}': client→gateway "
                        + "traffic carries the PII/secrets this gateway exists to protect and would be unencrypted.");
                    Console.Error.WriteLine(
                        "Put a TLS-terminating reverse proxy / sidecar (nginx, Caddy, Envoy, a cloud load balancer) in "
                        + "front and bind --host 127.0.0.1, OR pass --insecure if the hop to clients is already "
                        + "encrypted (service-mesh mTLS, TLS load balancer on a private network).");
                    return 1;
                }

                Console.WriteLine(
                    $"WARNING: --insecure — serving plaintext HTTP on '{host}'. Ensure the network hop to clients is "
                    + "encrypted (TLS proxy/sidecar or mesh); otherwise redacted-away secrets still travel in the clear.");
            }

            var key = Environment.GetEnvironmentVariable(keyEnv);
            if (string.IsNullOrEmpty(key))
            {
                Console.Error.WriteLine(
                    $"Upstream API key env var '{keyEnv}' is not set. The gateway holds the real upstream key so "
                    + $"clients never see it — set it first (e.g.  $env:{keyEnv}=\"sk-...\"  /  export {keyEnv}=sk-...).");
                return 1;
            }

            // Hardening: drop the secret from the process environment now that we hold it — so it can't be re-read
            // via /proc/<pid>/environ, inherited by child processes, or dumped from the env later.
            Environment.SetEnvironmentVariable(keyEnv, null);

            Redactor redactor;
            RedactionPolicy policy;
            string? configUpstream = null;
            IReadOnlyList<string> configClientKeys = [];
            var configScanResponses = false;

            if (!string.IsNullOrEmpty(configPath))
            {
                if (!File.Exists(configPath))
                {
                    Console.Error.WriteLine($"Config file not found: {configPath}");
                    return 1;
                }
                try
                {
                    (redactor, policy, configUpstream, configClientKeys, configScanResponses) = GatewayConfig.Load(configPath);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Failed to load gateway config '{configPath}': {ex.Message}");
                    return 1;
                }
                Console.WriteLine($"config: {Path.GetFullPath(configPath)}");
            }
            else
            {
                // Built-in: international + Polish (checksum-validated PESEL/NIP/REGON/IBAN) detectors, default policy.
                var intl = DefaultRedactionRules.All();
                var pl = PolishRedactionRules.All();
                var rules = new RedactionRule[intl.Length + pl.Length];
                intl.CopyTo(rules, 0);
                pl.CopyTo(rules, intl.Length);
                redactor = new Redactor(rules);
                policy = RedactionPolicy.Default();
                Console.WriteLine("policy: secrets (API/AWS keys, JWT, private key) BLOCK · PII (email, card+Luhn, SSN, IPv4, PL PESEL/NIP/REGON/IBAN) REDACT");
            }

            // CLI --upstream overrides the config's upstream.
            var resolvedUpstream = !string.IsNullOrEmpty(upstream) ? upstream : configUpstream;
            if (string.IsNullOrEmpty(resolvedUpstream))
            {
                Console.Error.WriteLine("No upstream set. Pass --upstream <url> or set \"upstream\" in --config.");
                return 1;
            }

            // Client keys: env var (preferred for secrets — comma/whitespace separated) merged with any in --config.
            var clientKeys = new List<string>(configClientKeys);
            var keysFromEnv = Environment.GetEnvironmentVariable(clientKeysEnv);
            if (!string.IsNullOrEmpty(keysFromEnv))
            {
                foreach (var part in keysFromEnv.Split([',', ' ', ';', '\t'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                {
                    clientKeys.Add(part);
                }
                Environment.SetEnvironmentVariable(clientKeysEnv, null);
            }

            if (clientKeys.Count > 0)
            {
                Console.WriteLine($"client auth: ON ({clientKeys.Count} gateway key(s)) — callers must send 'Authorization: Bearer <key>'");
            }
            else
            {
                Console.WriteLine($"client auth: OFF — set ${clientKeysEnv} (or \"clientKeys\" in --config) before exposing the gateway.");
            }

            // Response scanning: enabled by the CLI flag OR the config.
            var resolvedScanResponses = scanResponses || configScanResponses;
            if (resolvedScanResponses)
            {
                Console.WriteLine("response scan: ON — model-generated secrets/PII masked on non-streaming responses (your own values still restored).");
            }

            using var audit = new JsonLinesAuditSink(auditPath);
            Console.WriteLine($"audit log: {Path.GetFullPath(auditPath)}");
            RedactionGateway.Serve(host, port, resolvedUpstream, key, redactor, audit, policy, clientKeys, resolvedScanResponses);
            return 0;
        }

        // True for the loopback interface (127.0.0.0/8, localhost, ::1) — safe to serve plaintext HTTP on, because a
        // TLS-terminating proxy/sidecar fronts it. Anything else is a network-exposed bind and needs --insecure.
        private static bool IsLoopbackHost(string host)
        {
            return host is "localhost" or "::1" or "[::1]"
                || host.StartsWith("127.", StringComparison.Ordinal);
        }

        // ── score: run a trained XGBoost model (JSON) over a CSV of feature rows, pure-managed + zero-egress. ──
        public static int Score(string modelPath, string inputPath, string? outputPath, bool margin)
        {
            if (!File.Exists(modelPath))
            {
                Console.Error.WriteLine($"Model file not found: {modelPath}");
                Console.Error.WriteLine("Pass the path to an XGBoost model saved as JSON (booster.save_model(\"model.json\")).");
                return 1;
            }

            if (!File.Exists(inputPath))
            {
                Console.Error.WriteLine($"Input CSV not found: {inputPath}");
                return 1;
            }

            BoostedTreeModel model;
            try
            {
                model = XgboostModelLoader.Load(modelPath);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed to load '{Path.GetFileName(modelPath)}' as an XGBoost JSON model: {ex.Message}");
                return 1;
            }

            var rows = new List<float[]>();
            var lineNumber = 0;
            foreach (var raw in File.ReadLines(inputPath))
            {
                lineNumber++;
                var line = raw.Trim();
                if (line.Length == 0)
                {
                    continue;
                }

                var cells = line.Split(',');

                // A first row that does not parse fully as numbers is treated as a header and skipped.
                if (rows.Count == 0 && lineNumber == 1 && !LooksNumeric(cells))
                {
                    continue;
                }

                if (cells.Length != model.NumFeatures)
                {
                    Console.Error.WriteLine(
                        $"Line {lineNumber}: expected {model.NumFeatures} feature columns, got {cells.Length}.");
                    return 1;
                }

                var row = new float[model.NumFeatures];
                for (var c = 0; c < cells.Length; c++)
                {
                    if (!TryParseCell(cells[c], out row[c]))
                    {
                        Console.Error.WriteLine($"Line {lineNumber}, column {c + 1}: '{cells[c].Trim()}' is not a number.");
                        return 1;
                    }
                }

                rows.Add(row);
            }

            if (rows.Count == 0)
            {
                Console.Error.WriteLine("No feature rows found in the input CSV.");
                return 1;
            }

            // Flatten and score in one batch through the parallel, zero-allocation predictor.
            var groups = model.NumGroups;
            var flat = new float[(long)rows.Count * model.NumFeatures];
            for (var r = 0; r < rows.Count; r++)
            {
                rows[r].CopyTo(flat.AsSpan(r * model.NumFeatures, model.NumFeatures));
            }

            var outputs = new float[(long)rows.Count * groups];
            var sw = ValueStopwatch.StartNew();
            if (margin)
            {
                // Raw, pre-transform margins (XGBoost output_margin=True).
                for (var r = 0; r < rows.Count; r++)
                {
                    model.PredictRawMargins(rows[r], outputs.AsSpan(r * groups, groups));
                }
            }
            else
            {
                model.PredictBatchParallel(flat, rows.Count, outputs);
            }

            using var writer = outputPath is null ? null : new StreamWriter(outputPath);
            void Emit(string text)
            {
                if (writer is null)
                {
                    Console.Out.WriteLine(text);
                }
                else
                {
                    writer.WriteLine(text);
                }
            }

            Emit(HeaderLine(groups, margin));
            for (var r = 0; r < rows.Count; r++)
            {
                Emit(FormatRow(outputs.AsSpan(r * groups, groups)));
            }

            var nsPerRow = sw.GetElapsedTime().TotalMilliseconds * 1e6 / rows.Count;
            Console.Error.WriteLine(
                $"Scored {rows.Count:N0} rows · {model.NumTrees} trees · {model.Objective} · "
                + $"{model.NumGroups} output(s) · {sw.GetElapsedTime().TotalMilliseconds:F1} ms ({nsPerRow:F0} ns/row)"
                + (outputPath is null ? string.Empty : $" → {outputPath}"));
            return 0;
        }

        private static bool LooksNumeric(string[] cells)
        {
            foreach (var cell in cells)
            {
                if (!TryParseCell(cell, out _))
                {
                    return false;
                }
            }
            return true;
        }

        private static bool TryParseCell(string cell, out float value)
        {
            var token = cell.Trim();
            if (token.Length == 0
                || token.Equals("nan", StringComparison.OrdinalIgnoreCase)
                || token.Equals("na", StringComparison.OrdinalIgnoreCase)
                || token.Equals("null", StringComparison.OrdinalIgnoreCase)
                || token == "?")
            {
                // Missing value — XGBoost routes it via each node's default direction.
                value = float.NaN;
                return true;
            }

            return float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        private static string HeaderLine(int groups, bool margin)
        {
            if (groups == 1)
            {
                return margin ? "margin" : "prediction";
            }

            var prefix = margin ? "margin_" : "p_";
            var sb = new System.Text.StringBuilder();
            for (var g = 0; g < groups; g++)
            {
                if (g > 0)
                {
                    sb.Append(',');
                }
                sb.Append(prefix).Append(g.ToString(CultureInfo.InvariantCulture));
            }
            return sb.ToString();
        }

        private static string FormatRow(ReadOnlySpan<float> values)
        {
            if (values.Length == 1)
            {
                return values[0].ToString(CultureInfo.InvariantCulture);
            }

            var sb = new System.Text.StringBuilder();
            for (var i = 0; i < values.Length; i++)
            {
                if (i > 0)
                {
                    sb.Append(',');
                }
                sb.Append(values[i].ToString(CultureInfo.InvariantCulture));
            }
            return sb.ToString();
        }

        // Dominant weight-tensor quant: the type most of the model's bytes are in. Q4_K_M files mix Q4_K + Q6_K,
        // so we report the dominant type and flag the mix, plus the fraction of weight tensors that are quantized.
        private static (string Quant, bool Mixed, double QuantizedFraction) DescribeQuant(GgufReader reader)
        {
            var counts = new Dictionary<GgmlType, int>();
            var total = 0;
            var quantized = 0;
            foreach (var t in reader.Tensors.Values)
            {
                if (!t.Name.EndsWith(".weight", StringComparison.Ordinal))
                {
                    continue;
                }
                total++;
                counts[t.Type] = counts.GetValueOrDefault(t.Type) + 1;
                if (t.Type is not (GgmlType.F32 or GgmlType.F16 or GgmlType.BF16))
                {
                    quantized++;
                }
            }

            if (total == 0)
            {
                return ("unknown", false, 0d);
            }

            var dominant = GgmlType.F32;
            var best = -1;
            foreach (var (type, count) in counts)
            {
                if (count > best)
                {
                    best = count;
                    dominant = type;
                }
            }

            var distinctQuant = 0;
            foreach (var type in counts.Keys)
            {
                if (type is not (GgmlType.F32 or GgmlType.F16 or GgmlType.BF16))
                {
                    distinctQuant++;
                }
            }

            return (dominant.ToString(), distinctQuant > 1, (double)quantized / total);
        }

        private static string DescribeTokenizer(string ggmlModel) => ggmlModel switch
        {
            "gpt2" => "BPE (GPT-2 byte-level)",
            "llama" => "SentencePiece / BPE (Llama-family)",
            "bert" => "WordPiece (BERT)",
            "?" => "not declared",
            _ => ggmlModel,
        };

        // Mirrors the validated arch list in the README "Supported model families". Metadata-driven loader, so
        // an unknown arch may still load — we say so honestly rather than claim a hard yes/no.
        private static (string Supported, string Note) DescribeSupport(string arch, int experts) => arch switch
        {
            "qwen2" or "qwen3" or "llama" or "phi3" or "gemma2"
                => ("yes", "validated coherent on real models"),
            "qwen2moe" => ("yes", "MoE validated coherent (Qwen1.5-MoE)"),
            "gemma" or "gemma3" or "qwen3moe" or "command-r" or "deepseek2"
                => ("not yet", "architecture not implemented — will throw a clear NotSupportedException on load"),
            _ when experts > 0 => ("likely", "MoE arch — Mixtral-style loads; unvalidated for this exact arch, try it"),
            _ => ("unknown", "metadata-driven loader will attempt it; not on the validated list — try it"),
        };

        private static List<string> CollectWarnings(string arch, bool hasTemplate, string supported, string tokModel)
        {
            var warnings = new List<string>();
            if (!hasTemplate)
            {
                warnings.Add("no chat template in the GGUF — multi-turn formatting uses a generic fallback; quality may suffer.");
            }
            if (tokModel == "?")
            {
                warnings.Add("tokenizer model not declared in metadata — tokenization may be wrong.");
            }
            if (supported == "not yet")
            {
                warnings.Add($"architecture '{arch}' is not implemented yet — loading will fail.");
            }
            warnings.Add("RAG needs a separate embedding model — pass `--embed-model` to `overfit serve` (e.g. `overfit pull minilm`).");
            return warnings;
        }

        /// <summary>
        /// `overfit mcp &lt;model&gt; [--rag-dir d] [--whisper-model w]` — an MCP (Model Context
        /// Protocol) stdio server for hosts like Claude Code / Claude Desktop. stdout IS the
        /// protocol channel (newline-delimited JSON-RPC), so every status/log line goes to stderr.
        /// Tools: `ask` always; `rag_query` when --rag-dir is given (indexed up-front, embeddings
        /// from the chat model itself); `transcribe` when --whisper-model is given (loaded lazily
        /// on first call). Runs until the host closes our stdin.
        /// </summary>
        public static int Mcp(string model, string? ragDir, string? whisperModel)
        {
            var path = ModelCache.Resolve(model);
            if (path is null)
            {
                Console.Error.WriteLine($"Model '{model}' not found in {ModelCache.Dir}.");
                Console.Error.WriteLine($"Download it first:  overfit pull {model}   (or pass a .gguf path directly)");
                return 1;
            }

            if (!string.IsNullOrWhiteSpace(whisperModel) && !File.Exists(whisperModel))
            {
                Console.Error.WriteLine($"Whisper model not found: {whisperModel}");
                return 1;
            }

            Console.Error.WriteLine($"[overfit-mcp] loading {Path.GetFileName(path)} ...");
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

            try
            {
                client.AddSystem(DefaultSystemPrompt);
                var tools = new List<McpTool>
                {
                    OverfitMcpTools.CreateAsk(client),
                };

                if (!string.IsNullOrWhiteSpace(ragDir))
                {
                    Console.Error.WriteLine($"[overfit-mcp] indexing documents under {ragDir} ...");
                    var index = McpRagIndex.Build(client, ragDir, Console.Error);
                    Console.Error.WriteLine($"[overfit-mcp] rag_query ready ({index.ChunkCount} chunk(s)).");
                    tools.Add(OverfitMcpTools.CreateRagQuery(index));
                }

                if (!string.IsNullOrWhiteSpace(whisperModel))
                {
                    var ggml = whisperModel;
                    tools.Add(OverfitMcpTools.CreateTranscribe(() =>
                    {
                        Console.Error.WriteLine($"[overfit-mcp] loading Whisper {Path.GetFileName(ggml)} ...");
                        return WhisperTranscriber.Load(ggml);
                    }));
                }

                var version = typeof(Commands).Assembly.GetName().Version?.ToString(3) ?? "0.0.0";
                var server = new McpServer("overfit", version, tools, Console.Error);
                Console.Error.WriteLine($"[overfit-mcp] serving {tools.Count} tool(s) over stdio ({Path.GetFileNameWithoutExtension(path)}). The host stops the server by closing stdin.");
                server.Run(Console.In, Console.Out);
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[overfit-mcp] startup failed: {ex.Message}");
                return 1;
            }
            finally
            {
                client.Dispose();
            }
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

            var envDir = Environment.GetEnvironmentVariable(OverfitEnvironment.OrpheusDir);
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
                Environment.GetEnvironmentVariable(OverfitEnvironment.SnacDir),
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
