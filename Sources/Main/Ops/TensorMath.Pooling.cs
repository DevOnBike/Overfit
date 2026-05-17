// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
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

            var inputSize = channels * h * w;
            var outputSize = channels * oH * oW;

            if (batchSize < BatchSequentialThreshold)
            {
                for (var n = 0; n < batchSize; n++)
                {
                    PoolingKernels.MaxPool2DForwardWithIndicesNchw(
                        input.DataView.AsReadOnlySpan().Slice(n * inputSize, inputSize),
                        output.DataView.AsSpan().Slice(n * outputSize, outputSize),
                        maxIndices.DataView.AsSpan().Slice(n * outputSize, outputSize),
                        channels, h, w, pool,
                        batchOffset: n * inputSize);
                }
            }
            else
            {
                var inputSpan = input.DataView.AsReadOnlySpan();
                var outputSpan = output.DataView.AsSpan();
                var indicesSpan = maxIndices.DataView.AsSpan();

                unsafe
                {
                    fixed (float* inPtr = inputSpan, outPtr = outputSpan)
                    fixed (float* idxPtr = indicesSpan)
                    {
                        var ctx = new MaxPool2DForwardCtx
                        {
                            Input = inPtr,
                            Output = outPtr,
                            Indices = idxPtr,
                            InputSize = inputSize,
                            OutputSize = outputSize,
                            Channels = channels,
                            H = h,
                            W = w,
                            Pool = pool,
                        };
                        OverfitParallelFor.For(0, batchSize, &MaxPool2DForwardChunk, &ctx);
                    }
                }
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
                output.DataView.Size < ParallelThreshold)
            {
                var oGS = output.GradView.AsReadOnlySpan();
                var iGS = input.GradView.AsSpan();
                var idxS = maxIndices.DataView.AsReadOnlySpan();

                for (var i = 0; i < oGS.Length; i++)
                {
                    iGS[(int)idxS[i]] += oGS[i];
                }

                return;
            }

            var oGSpan = output.GradView.AsReadOnlySpan();
            var iGSpan = input.GradView.AsSpan();
            var idxSpan = maxIndices.DataView.AsReadOnlySpan();

            unsafe
            {
                fixed (float* oGPtr = oGSpan, iGPtr = iGSpan, idxPtr = idxSpan)
                {
                    var ctx = new MaxPool2DBackwardCtx
                    {
                        GradOutput = oGPtr,
                        GradInput = iGPtr,
                        Indices = idxPtr,
                        OutputPerBatch = outputPerBatch,
                    };
                    OverfitParallelFor.For(0, batchSize, &MaxPool2DBackwardChunk, &ctx);
                }
            }
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
            var batchSize = input.Shape.D0;
            var spatialSize = h * w;
            var scale = 1f / spatialSize;

            var output = AllocateNode(
                graph,
                new TensorShape(batchSize, channels),
                input.RequiresGrad,
                clearMemory: false);

            var inS = input.DataView.AsReadOnlySpan();
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
                unsafe
                {
                    fixed (float* inPtr = inS, outPtr = outS)
                    {
                        var ctx = new GlobalAvgPool2DForwardCtx
                        {
                            Input = inPtr,
                            Output = outPtr,
                            Channels = channels,
                            SpatialSize = spatialSize,
                            Scale = scale,
                        };
                        OverfitParallelFor.For(0, batchSize, &GlobalAvgPool2DForwardChunk, &ctx);
                    }
                }
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

            var batchSize = input.Shape.D0;
            var spatialSize = h * w;
            var scale = 1f / spatialSize;

            if (batchSize < BatchSequentialThreshold ||
                (long)batchSize * channels * spatialSize < ParallelThreshold)
            {
                var oGS = output.GradView.AsReadOnlySpan();
                var iGS = input.GradView.AsSpan();

                for (var n = 0; n < batchSize; n++)
                {
                    for (var c = 0; c < channels; c++)
                    {
                        var grad = oGS[n * channels + c] * scale;
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

            var oGSpan = output.GradView.AsReadOnlySpan();
            var iGSpan = input.GradView.AsSpan();

            unsafe
            {
                fixed (float* oGPtr = oGSpan, iGPtr = iGSpan)
                {
                    var ctx = new GlobalAvgPool2DBackwardCtx
                    {
                        GradOutput = oGPtr,
                        GradInput = iGPtr,
                        Channels = channels,
                        SpatialSize = spatialSize,
                        Scale = scale,
                    };
                    OverfitParallelFor.For(0, batchSize, &GlobalAvgPool2DBackwardChunk, &ctx);
                }
            }
        }

        // ── OverfitParallelFor chunk bodies + contexts ────────────────────────

        private unsafe struct MaxPool2DForwardCtx
        {
            public float* Input;
            public float* Output;
            public float* Indices;
            public int InputSize;
            public int OutputSize;
            public int Channels;
            public int H;
            public int W;
            public int Pool;
        }

        private static unsafe void MaxPool2DForwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<MaxPool2DForwardCtx>(contextPtr);
            for (var n = chunkStart; n < chunkEnd; n++)
            {
                var inputSlice = new ReadOnlySpan<float>(ctx.Input + n * ctx.InputSize, ctx.InputSize);
                var outputSlice = new Span<float>(ctx.Output + n * ctx.OutputSize, ctx.OutputSize);
                var indicesSlice = new Span<float>(ctx.Indices + n * ctx.OutputSize, ctx.OutputSize);

                PoolingKernels.MaxPool2DForwardWithIndicesNchw(
                    inputSlice,
                    outputSlice,
                    indicesSlice,
                    ctx.Channels, ctx.H, ctx.W, ctx.Pool,
                    batchOffset: n * ctx.InputSize);
            }
        }

        private unsafe struct MaxPool2DBackwardCtx
        {
            public float* GradOutput;
            public float* GradInput;
            public float* Indices;
            public int OutputPerBatch;
        }

        private static unsafe void MaxPool2DBackwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<MaxPool2DBackwardCtx>(contextPtr);
            for (var n = chunkStart; n < chunkEnd; n++)
            {
                var start = n * ctx.OutputPerBatch;
                var end = start + ctx.OutputPerBatch;
                for (var i = start; i < end; i++)
                {
                    ctx.GradInput[(int)ctx.Indices[i]] += ctx.GradOutput[i];
                }
            }
        }

        private unsafe struct GlobalAvgPool2DForwardCtx
        {
            public float* Input;
            public float* Output;
            public int Channels;
            public int SpatialSize;
            public float Scale;
        }

        private static unsafe void GlobalAvgPool2DForwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<GlobalAvgPool2DForwardCtx>(contextPtr);
            for (var n = chunkStart; n < chunkEnd; n++)
            {
                for (var c = 0; c < ctx.Channels; c++)
                {
                    var channelSpan = new ReadOnlySpan<float>(
                        ctx.Input + n * ctx.Channels * ctx.SpatialSize + c * ctx.SpatialSize,
                        ctx.SpatialSize);
                    ctx.Output[n * ctx.Channels + c] = TensorPrimitives.Sum(channelSpan) * ctx.Scale;
                }
            }
        }

        private unsafe struct GlobalAvgPool2DBackwardCtx
        {
            public float* GradOutput;
            public float* GradInput;
            public int Channels;
            public int SpatialSize;
            public float Scale;
        }

        private static unsafe void GlobalAvgPool2DBackwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<GlobalAvgPool2DBackwardCtx>(contextPtr);
            for (var n = chunkStart; n < chunkEnd; n++)
            {
                for (var c = 0; c < ctx.Channels; c++)
                {
                    var grad = ctx.GradOutput[n * ctx.Channels + c] * ctx.Scale;
                    var offset = n * ctx.Channels * ctx.SpatialSize + c * ctx.SpatialSize;
                    for (var i = 0; i < ctx.SpatialSize; i++)
                    {
                        ctx.GradInput[offset + i] += grad;
                    }
                }
            }
        }
    }
}
