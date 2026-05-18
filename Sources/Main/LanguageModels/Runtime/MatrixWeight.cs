// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// A decode-time weight matrix resident either as F32
    /// (<see cref="TensorStorage{T}"/> of <see cref="float"/> or a raw
    /// <see cref="float"/>[]) or as FP16 (<see cref="TensorStorage{T}"/> of
    /// <see cref="Half"/>) — Slot 2c.
    ///
    /// FP16-resident weights halve the DRAM traffic of the memory-bandwidth-bound
    /// decode matmul; <see cref="SingleTokenProjectionKernel.ProjectHalf"/> widens
    /// each weight row to F32 one L1-resident tile at a time.
    ///
    /// All three backings convert implicitly, so existing F32 construction sites
    /// (GPT-1 / GPT-2, the binary loader, test code passing raw arrays) compile and
    /// behave unchanged — only the GGUF loader chooses FP16.
    /// </summary>
    public readonly struct MatrixWeight
    {
        private readonly TensorStorage<float>? _f32;
        private readonly TensorStorage<Half>? _f16;
        private readonly float[]? _f32Array;

        public MatrixWeight(TensorStorage<float> f32)
        {
            _f32 = f32 ?? throw new ArgumentNullException(nameof(f32));
            _f16 = null;
            _f32Array = null;
        }

        public MatrixWeight(TensorStorage<Half> f16)
        {
            _f16 = f16 ?? throw new ArgumentNullException(nameof(f16));
            _f32 = null;
            _f32Array = null;
        }

        public MatrixWeight(float[] f32Array)
        {
            _f32Array = f32Array ?? throw new ArgumentNullException(nameof(f32Array));
            _f32 = null;
            _f16 = null;
        }

        public static implicit operator MatrixWeight(TensorStorage<float> f32) => new(f32);

        public static implicit operator MatrixWeight(TensorStorage<Half> f16) => new(f16);

        public static implicit operator MatrixWeight(float[] f32Array) => new(f32Array);

        /// <summary>True when the weights are resident as FP16.</summary>
        public bool IsHalf => _f16 is not null;

        /// <summary>Element count, regardless of backing.</summary>
        public int Length => _f16?.Length ?? _f32Array?.Length ?? _f32!.Length;

        public bool IsEmpty => Length == 0;

        /// <summary>F32 view. Valid only when <see cref="IsHalf"/> is false.</summary>
        public ReadOnlySpan<float> F32 => _f32Array is not null ? _f32Array : _f32!.AsReadOnlySpan();

        /// <summary>FP16 view. Valid only when <see cref="IsHalf"/> is true.</summary>
        public ReadOnlySpan<Half> F16 => _f16!.AsReadOnlySpan();

        /// <summary>
        /// Underlying F32 storage, or null when FP16- or array-backed. Used by the
        /// in-place LoRA weight-merge path, which operates only on F32 storage.
        /// </summary>
        public TensorStorage<float>? F32Storage => _f32;

        /// <summary>Disposes whichever <see cref="TensorStorage{T}"/> backing is present.</summary>
        public void Dispose()
        {
            _f32?.Dispose();
            _f16?.Dispose();
        }
    }
}
