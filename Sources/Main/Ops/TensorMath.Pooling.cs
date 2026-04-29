// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // MAX POOL 2D
        // ====================================================================

        public static AutogradNode MaxPool2D(
            ComputationGraph graph,
            AutogradNode input,
            int channels,
            int h,
            int w,
            int pool)
        {
            var batchSize = input.Shape.D0;
            int oH = h / pool, oW = w / pool;
            var needsIndices = input.RequiresGrad;

            var output = AllocateNode(
                graph,
                new TensorShape(batchSize, channels, oH, oW),
                needsIndices,
                clearMemory: false);

            if (!needsIndices)
            {
                // Inference path: no index tracking, no maxIndices allocation.
                // Dispatches to SIMD-friendly pool=2 fast path inside PoolingKernels.
                PoolingKernels.MaxPool2DForwardNchw(
                    input.DataView.AsReadOnlySpan(),
                    output.DataView.AsSpan(),
                    channels, h, w, pool);

                return output;
            }

            // Training path: record which input pixel won each pool window so
            // that backward can scatter gradients in O(output) time without re-scan.
            var maxIndices = AllocateNode(
                graph,
                new TensorShape(batchSize, channels, oH, oW),
                requiresGrad: false,
                clearMemory: false);

            var inputSize  = channels * h * w;
            var outputSize = channels * oH * oW;

            if (batchSize < BatchSequentialThreshold)
            {
                for (var n = 0; n < batchSize; n++)
                {
                    PoolingKernels.MaxPool2DForwardWithIndicesNchw(
                        input.DataView.AsReadOnlySpan().Slice(n * inputSize,  inputSize),
                        output.DataView.AsSpan()          .Slice(n * outputSize, outputSize),
                        maxIndices.DataView.AsSpan()      .Slice(n * outputSize, outputSize),
                        channels, h, w, pool,
                        batchOffset: n * inputSize);
                }
            }
            else
            {
                Parallel.For(
                    0,
                    batchSize,
                    OverfitParallel.Options,
                    n =>
                    {
                        PoolingKernels.MaxPool2DForwardWithIndicesNchw(
                            input.DataView.AsReadOnlySpan().Slice(n * inputSize,  inputSize),
                            output.DataView.AsSpan()          .Slice(n * outputSize, outputSize),
                            maxIndices.DataView.AsSpan()      .Slice(n * outputSize, outputSize),
                            channels, h, w, pool,
                            batchOffset: n * inputSize);
                    });
            }

            graph?.Record(OpCode.MaxPool2D, output, input, maxIndices);

            return output;
        }

        public static void MaxPool2DBackward(
            AutogradNode input,
            AutogradNode maxIndices,
            AutogradNode output)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var batchSize = input.Shape.D0;
            var outputPerBatch = output.DataView.Size / batchSize;

            if (batchSize < BatchSequentialThreshold ||
                (long)output.DataView.Size < ParallelThreshold)
            {
                var oGS  = output.GradView.AsReadOnlySpan();
                var iGS  = input.GradView.AsSpan();
                var idxS = maxIndices.DataView.AsReadOnlySpan();

                for (var i = 0; i < oGS.Length; i++)
                {
                    iGS[(int)idxS[i]] += oGS[i];
                }

                return;
            }

            Parallel.For(
                0,
                batchSize,
                OverfitParallel.Options,
                n =>
                {
                    var oGS  = output.GradView.AsReadOnlySpan();
                    var iGS  = input.GradView.AsSpan();
                    var idxS = maxIndices.DataView.AsReadOnlySpan();

                    var start = n * outputPerBatch;
                    var end   = start + outputPerBatch;

                    for (var i = start; i < end; i++)
                    {
                        iGS[(int)idxS[i]] += oGS[i];
                    }
                });
        }

        // ====================================================================
        // GLOBAL AVERAGE POOL 2D
        // ====================================================================

        public static AutogradNode GlobalAveragePool2D(
            ComputationGraph graph,
            AutogradNode input,
            int channels,
            int h,
            int w)
        {
            var batchSize   = input.Shape.D0;
            var spatialSize = h * w;
            var scale       = 1f / spatialSize;

            var output = AllocateNode(
                graph,
                new TensorShape(batchSize, channels),
                input.RequiresGrad,
                clearMemory: false);

            var inS  = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();

            if (batchSize < BatchSequentialThreshold)
            {
                for (var n = 0; n < batchSize; n++)
                {
                    for (var c = 0; c < channels; c++)
                    {
                        outS[n * channels + c] =
                            TensorPrimitives.Sum(
                                inS.Slice(n * channels * spatialSize + c * spatialSize, spatialSize))
                            * scale;
                    }
                }
            }
            else
            {
                Parallel.For(
                    0,
                    batchSize,
                    OverfitParallel.Options,
                    n =>
                    {
                        for (var c = 0; c < channels; c++)
                        {
                            output.DataView.AsSpan()[n * channels + c] =
                                TensorPrimitives.Sum(
                                    input.DataView.AsReadOnlySpan()
                                        .Slice(n * channels * spatialSize + c * spatialSize, spatialSize))
                                * scale;
                        }
                    });
            }

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.GlobalAveragePool2D, output, input, null, h, w, channels);
            }

            return output;
        }

        public static void GlobalAvgPool2DBackward(
            AutogradNode input,
            AutogradNode output,
            int h,
            int w,
            int channels)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var batchSize   = input.Shape.D0;
            var spatialSize = h * w;
            var scale       = 1f / spatialSize;

            if (batchSize < BatchSequentialThreshold ||
                (long)batchSize * channels * spatialSize < ParallelThreshold)
            {
                var oGS = output.GradView.AsReadOnlySpan();
                var iGS = input.GradView.AsSpan();

                for (var n = 0; n < batchSize; n++)
                {
                    for (var c = 0; c < channels; c++)
                    {
                        var grad         = oGS[n * channels + c] * scale;
                        var channelSlice = iGS.Slice(
                            n * channels * spatialSize + c * spatialSize,
                            spatialSize);

                        for (var i = 0; i < spatialSize; i++)
                        {
                            channelSlice[i] += grad;
                        }
                    }
                }

                return;
            }

            Parallel.For(
                0,
                batchSize,
                OverfitParallel.Options,
                n =>
                {
                    var oGS = output.GradView.AsReadOnlySpan();
                    var iGS = input.GradView.AsSpan();

                    for (var c = 0; c < channels; c++)
                    {
                        var grad         = oGS[n * channels + c] * scale;
                        var channelSlice = iGS.Slice(
                            n * channels * spatialSize + c * spatialSize,
                            spatialSize);

                        for (var i = 0; i < spatialSize; i++)
                        {
                            channelSlice[i] += grad;
                        }
                    }
                });
        }
    }
}
