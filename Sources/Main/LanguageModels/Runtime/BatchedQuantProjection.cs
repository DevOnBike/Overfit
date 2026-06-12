// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Batched (prefill) projection dispatch over a <see cref="DecodeWeight"/>: picks the
    /// <c>ProjectBatched</c> kernel matching the weight's resident format (Q6_K / Q4_K / Q8_0 / F32)
    /// and runs <c>rows</c> activation rows × the weight matrix in one pass — each weight
    /// row read from DRAM once, reused across all rows (the prefill weight-bandwidth amortisation).
    /// Activation-quantization scratch is POOLED per call and handed to the kernels as exact-length
    /// slices (so any kernel-side <c>.Length</c> arithmetic is unchanged). This dispatcher runs once
    /// per projection per layer per prefill — it was the single largest allocator of the prefill path.
    /// </summary>
    internal static class BatchedQuantProjection
    {
        public static void Dispatch(
            ReadOnlySpan<float> input,
            int rows,
            in DecodeWeight weight,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            if (weight.IsQ6K)
            {
                var w = weight.Quantized6K;
                var spr = w.SuperBlocksPerRow;
                var groups = rows * spr * Q6KDotKernel.GroupsPerSuperBlock;
                var qBytes = PooledBuffer<sbyte>.RentArray(rows * inputSize);
                var scales = PooledBuffer<float>.RentArray(rows * spr);
                var sums = PooledBuffer<short>.RentArray(groups);
                try
                {
                    Q6KDotKernel.ProjectBatched(
                        input, rows, w, bias, output,
                        qBytes.AsSpan(0, rows * inputSize), scales.AsSpan(0, rows * spr),
                        sums.AsSpan(0, groups));
                }
                finally
                {
                    PooledBuffer<short>.ReturnArray(sums);
                    PooledBuffer<float>.ReturnArray(scales);
                    PooledBuffer<sbyte>.ReturnArray(qBytes);
                }
            }
            else if (weight.IsQ4K)
            {
                var w = weight.Quantized4K;
                var spr = w.SuperBlocksPerRow;
                var groups = rows * spr * Q4KDotKernel.GroupsPerSuperBlock;
                var qBytes = PooledBuffer<sbyte>.RentArray(rows * inputSize);
                var scales = PooledBuffer<float>.RentArray(rows * spr);
                var sums = PooledBuffer<short>.RentArray(groups);
                try
                {
                    Q4KDotKernel.ProjectBatched(
                        input, rows, w, bias, output,
                        qBytes.AsSpan(0, rows * inputSize), scales.AsSpan(0, rows * spr),
                        sums.AsSpan(0, groups));
                }
                finally
                {
                    PooledBuffer<short>.ReturnArray(sums);
                    PooledBuffer<float>.ReturnArray(scales);
                    PooledBuffer<sbyte>.ReturnArray(qBytes);
                }
            }
            else if (weight.IsQuantized)
            {
                var w = weight.Quantized;
                var bpr = inputSize / Q8DotKernel.BlockSize;
                var qBytes = PooledBuffer<sbyte>.RentArray(rows * inputSize);
                var scales = PooledBuffer<float>.RentArray(rows * bpr);
                try
                {
                    Q8DotKernel.ProjectBatched(
                        input, rows, w, bias, output,
                        qBytes.AsSpan(0, rows * inputSize), scales.AsSpan(0, rows * bpr));
                }
                finally
                {
                    PooledBuffer<float>.ReturnArray(scales);
                    PooledBuffer<sbyte>.ReturnArray(qBytes);
                }
            }
            else
            {
                BatchedProjectionKernel.Project(input, rows, weight.F32, bias, output, inputSize, outputSize);
            }
        }
    }
}
