// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
    /// <summary>
    /// Whisper hyper-parameters (the 10 ints + float flag at the head of a whisper.cpp ggml file).
    /// </summary>
    public sealed record WhisperConfig(
        int NVocab,
        int NAudioCtx,
        int NAudioState,
        int NAudioHead,
        int NAudioLayer,
        int NTextCtx,
        int NTextState,
        int NTextHead,
        int NTextLayer,
        int NMels,
        bool F16)
    {
        /// <summary>Multilingual models carry the language tokens (n_vocab ≥ 51865); English-only = 51864.</summary>
        public bool IsMultilingual => NVocab >= 51865;

        /// <summary>Number of language tokens (whisper.cpp: <c>n_vocab − 51765 − (multilingual ? 1 : 0)</c>).</summary>
        public int NumLanguages => NVocab - 51765 - (IsMultilingual ? 1 : 0);
    }

    /// <summary>One loaded weight tensor: dequantized F32 data in logical (un-reversed) shape order.</summary>
    public sealed class WhisperTensor
    {
        public WhisperTensor(int[] shape, float[] data)
        {
            Shape = shape;
            Data = data;
        }

        /// <summary>Logical dimensions (ggml stores them reversed; the loader un-reverses).</summary>
        public int[] Shape { get; }

        /// <summary>F32 weight values (row-major over <see cref="Shape"/>).</summary>
        public float[] Data { get; }

        public long ElementCount => Data.Length;
    }

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

        public WhisperConfig Config { get; }

        /// <summary>Mel filterbank rows (<c>n_mel</c>, e.g. 80).</summary>
        public int MelFilterRows { get; }

        /// <summary>Mel filterbank cols (<c>1 + n_fft/2</c>, e.g. 201).</summary>
        public int MelFilterCols { get; }

        /// <summary>The model's own mel filterbank <c>[MelFilterRows × MelFilterCols]</c> — use for bit-parity
        /// with whisper.cpp instead of computing the Slaney filterbank.</summary>
        public float[] MelFilters { get; }

        /// <summary>Token id → byte-level BPE string (as stored in the file).</summary>
        public IReadOnlyList<string> Vocab { get; }

        /// <summary>Weight tensors by name.</summary>
        public IReadOnlyDictionary<string, WhisperTensor> Tensors { get; }
    }
}
