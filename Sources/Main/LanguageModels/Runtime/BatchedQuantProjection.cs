// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Batched (prefill) projection dispatch over a <see cref="DecodeWeight"/>: picks the
    /// <c>ProjectBatched</c> kernel matching the weight's resident format (Q6_K / Q4_K / Q8_0 / F32)
    /// and runs <c>rows</c> activation rows × the weight matrix in one pass — each weight
    /// row read from DRAM once, reused across all rows (the prefill weight-bandwidth amortisation).
    /// Activation-quantization scratch is allocated per call (prefill is a one-time pass, not the
    /// zero-allocation decode hot path). Shared by the FFN and attention batched paths.
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
                Q6KDotKernel.ProjectBatched(
                    input, rows, w, bias, output,
                    new sbyte[rows * inputSize], new float[rows * spr],
                    new short[rows * spr * Q6KDotKernel.GroupsPerSuperBlock]);
            }
            else if (weight.IsQ4K)
            {
                var w = weight.Quantized4K;
                var spr = w.SuperBlocksPerRow;
                Q4KDotKernel.ProjectBatched(
                    input, rows, w, bias, output,
                    new sbyte[rows * inputSize], new float[rows * spr],
                    new short[rows * spr * Q4KDotKernel.GroupsPerSuperBlock]);
            }
            else if (weight.IsQuantized)
            {
                var w = weight.Quantized;
                var bpr = inputSize / Q8DotKernel.BlockSize;
                Q8DotKernel.ProjectBatched(
                    input, rows, w, bias, output,
                    new sbyte[rows * inputSize], new float[rows * bpr]);
            }
            else
            {
                BatchedProjectionKernel.Project(input, rows, weight.F32, bias, output, inputSize, outputSize);
            }
        }
    }
}
