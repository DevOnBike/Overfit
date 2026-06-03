// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
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
}
