// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tokenization;

namespace DevOnBike.Overfit.LanguageModels
{
    /// <summary>
    /// User-facing GPT-2 entry point. Composes <see cref="GPT1Model"/> +
    /// <see cref="CachedSlmInferenceEngine"/> + <see cref="BytePairEncoder"/>
    /// behind a single disposable handle, so the common case is three lines:
    ///
    /// <code>
    /// using var gpt2    = Gpt2.LoadSmall(@"C:\gpt2");
    /// using var session = gpt2.CreateSession();
    /// session.Reset(gpt2.Tokenizer.Encode("The future of software is"));
    /// </code>
    ///
    /// <para>
    /// Directory convention used by the per-size factories
    /// (<see cref="LoadSmall"/>, <see cref="LoadMedium"/>, <see cref="LoadLarge"/>,
    /// <see cref="LoadXL"/>): the directory contains
    /// <c>gpt2_&lt;size&gt;.bin</c>, <c>vocab.json</c>, and <c>merges.txt</c>.
    /// Convert these once with <c>Scripts/convert_gpt2.py --size &lt;size&gt; --out &lt;dir&gt;</c>.
    /// </para>
    ///
    /// <para>
    /// Power users that need a different file layout or tokenizer can drop
    /// down to the raw API (<c>new GPT1Model(Gpt2Config.Small)</c> +
    /// <c>CachedSlmInferenceEngine.FromGpt1</c>) — this class is convenience,
    /// not gatekeeping.
    /// </para>
    /// </summary>
    public sealed class Gpt2 : IDisposable
    {
        private readonly GPT1Model _model;
        private readonly CachedSlmInferenceEngine _engine;
        private readonly BytePairEncoder _tokenizer;
        private bool _disposed;

        /// <summary>Configuration this instance was loaded with (Small / Medium / Large / XL).</summary>
        public GPT1Config Config
        {
            get;
        }

        /// <summary>GPT-2 BPE tokenizer loaded from the supplied vocab.json + merges.txt.</summary>
        public ITokenizer Tokenizer => _tokenizer;

        private Gpt2(
            GPT1Model model,
            CachedSlmInferenceEngine engine,
            BytePairEncoder tokenizer,
            GPT1Config config)
        {
            _model = model;
            _engine = engine;
            _tokenizer = tokenizer;
            Config = config;
        }

        // ── Per-size convenience factories ────────────────────────────────────

        /// <summary>
        /// Loads GPT-2 Small (124M) from a directory containing
        /// <c>gpt2_small.bin</c>, <c>vocab.json</c>, <c>merges.txt</c>.
        /// </summary>
        public static Gpt2 LoadSmall(string modelDir) => LoadFromDir(modelDir, "small", Gpt2Config.Small);

        /// <summary>
        /// Loads GPT-2 Medium (355M) from a directory containing
        /// <c>gpt2_medium.bin</c>, <c>vocab.json</c>, <c>merges.txt</c>.
        /// </summary>
        public static Gpt2 LoadMedium(string modelDir) => LoadFromDir(modelDir, "medium", Gpt2Config.Medium);

        /// <summary>
        /// Loads GPT-2 Large (774M) from a directory containing
        /// <c>gpt2_large.bin</c>, <c>vocab.json</c>, <c>merges.txt</c>.
        /// </summary>
        public static Gpt2 LoadLarge(string modelDir) => LoadFromDir(modelDir, "large", Gpt2Config.Large);

        /// <summary>
        /// Loads GPT-2 XL (1.5B) from a directory containing
        /// <c>gpt2_xl.bin</c>, <c>vocab.json</c>, <c>merges.txt</c>.
        /// </summary>
        public static Gpt2 LoadXL(string modelDir) => LoadFromDir(modelDir, "xl", Gpt2Config.XL);

        // ── Full-control factory ──────────────────────────────────────────────

        /// <summary>
        /// Loads GPT-2 from explicit file paths and a configuration preset.
        /// Use this when the directory convention doesn't fit (e.g. shared
        /// tokenizer across multiple model sizes, custom file naming).
        /// </summary>
        public static Gpt2 Load(
            string modelPath,
            string vocabPath,
            string mergesPath,
            GPT1Config config)
        {
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"GPT-2 model not found: '{modelPath}'.", modelPath);
            }
            if (!File.Exists(vocabPath))
            {
                throw new FileNotFoundException($"GPT-2 vocab.json not found: '{vocabPath}'.", vocabPath);
            }
            if (!File.Exists(mergesPath))
            {
                throw new FileNotFoundException($"GPT-2 merges.txt not found: '{mergesPath}'.", mergesPath);
            }

            var model = new GPT1Model(config);
            try
            {
                model.Eval();
                using (var fs = File.OpenRead(modelPath))
                using (var br = new BinaryReader(fs))
                {
                    model.Load(br);
                }

                var engine = CachedSlmInferenceEngine.FromGpt1(model);
                try
                {
                    var tokenizer = BytePairEncoder.Load(vocabPath, mergesPath);
                    return new Gpt2(model, engine, tokenizer, config);
                }
                catch
                {
                    engine.Dispose();
                    throw;
                }
            }
            catch
            {
                model.Dispose();
                throw;
            }
        }

        // ── Public API ────────────────────────────────────────────────────────

        /// <summary>
        /// Creates a new inference session bound to the loaded model. Each session
        /// holds its own KV cache and per-token scratch — multiple sessions can
        /// share the underlying weights safely.
        /// </summary>
        public CachedSlmSession CreateSession()
        {
            ThrowIfDisposed();
            return _engine.CreateSession();
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;

            // Engine first — it holds references to weights/storage owned by the
            // model. Then the model itself.
            _engine.Dispose();
            _model.Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(Gpt2));
            }
        }

        private static Gpt2 LoadFromDir(string modelDir, string sizeSuffix, GPT1Config config)
        {
            if (string.IsNullOrWhiteSpace(modelDir))
            {
                throw new ArgumentException("modelDir must be a non-empty path.", nameof(modelDir));
            }

            return Load(
                modelPath: Path.Combine(modelDir, $"gpt2_{sizeSuffix}.bin"),
                vocabPath: Path.Combine(modelDir, "vocab.json"),
                mergesPath: Path.Combine(modelDir, "merges.txt"),
                config: config);
        }
    }
}
