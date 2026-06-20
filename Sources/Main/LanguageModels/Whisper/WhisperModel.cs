// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
    /// <summary>
    /// A fully loaded whisper.cpp ggml model: config, the mel filterbank baked into the file, the token
    /// vocabulary (byte-level BPE strings), and every weight tensor by name (<c>encoder.*</c> / <c>decoder.*</c>).
    /// </summary>
    public sealed class WhisperModel
    {
        public WhisperModel(
            WhisperConfig config,
            int melFilterRows,
            int melFilterCols,
            float[] melFilters,
            IReadOnlyList<string> vocab,
            IReadOnlyDictionary<string, WhisperTensor> tensors)
        {
            Config = config;
            MelFilterRows = melFilterRows;
            MelFilterCols = melFilterCols;
            MelFilters = melFilters;
            Vocab = vocab;
            Tensors = tensors;
        }

        public WhisperConfig Config
        {
            get;
        }

        /// <summary>Mel filterbank rows (<c>n_mel</c>, e.g. 80).</summary>
        public int MelFilterRows
        {
            get;
        }

        /// <summary>Mel filterbank cols (<c>1 + n_fft/2</c>, e.g. 201).</summary>
        public int MelFilterCols
        {
            get;
        }

        /// <summary>The model's own mel filterbank <c>[MelFilterRows × MelFilterCols]</c> — use for bit-parity
        /// with whisper.cpp instead of computing the Slaney filterbank.</summary>
        public float[] MelFilters
        {
            get;
        }

        /// <summary>Token id → byte-level BPE string (as stored in the file).</summary>
        public IReadOnlyList<string> Vocab
        {
            get;
        }

        /// <summary>Weight tensors by name.</summary>
        public IReadOnlyDictionary<string, WhisperTensor> Tensors
        {
            get;
        }
    }
}
