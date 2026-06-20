// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // CONV2D
        // ====================================================================

        public static AutogradNode Conv2D(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode weights,
            int inC,
            int outC,
            int h,
            int w,
            int k,
            int padding = 0,
            int stride = 1,
            AutogradNode bias = null)
        {
            var outH = (h + 2 * padding - k) / stride + 1;
            var outW = (w + 2 * padding - k) / stride + 1;
            var batchSize = input.Shape.D0;
            var kSqInC = k * k * inC;
            var spatialOut = outH * outW;
            var inputPlaneLength = inC * h * w;
            var outputPlaneLength = outC * spatialOut;

            var outputNode = AllocateNode(
                graph,
                new TensorShape(batchSize, outC, outH, outW),
                input.RequiresGrad || weights.RequiresGrad || (bias is not null && bias.RequiresGrad),
                clearMemory: true);

            // The convolution workspace holds per-worker im2col scratch buffers that the
            // parallel pass below writes into. Normally it lives on the graph and is reused
            // across all conv ops in a training step (zero per-call allocation). For
            // inference paths that pass graph=null (e.g. a lightweight forward over a
            // single image without recording a tape), we create a local workspace for the
            // duration of the call. The local path allocates but isn't on the training
            // hot path; it's the cost of decoupling inference from graph state.
            Conv2DWorkspace localWorkspace = null;
            Conv2DWorkspace workspace;

            if (graph is not null)
            {
                workspace = graph.GetConv2DWorkspace(batchSize, inC, outC, h, w, k, padding, stride);
            }
            else
            {
                localWorkspace = new Conv2DWorkspace();
                var localWorkers = Math.Max(1, Math.Min(OverfitParallel.MaxDegreeOfParallelism, Math.Max(1, batchSize)));
                localWorkspace.Ensure(
                    localWorkers,
                    colLength: kSqInC * spatialOut,
                    partialWeightGradientLength: outC * kSqInC);
                workspace = localWorkspace;
            }

            try
            {
                var workerCount = workspace.WorkerCount;

                // Each worker owns a private col-buffer slice and strides the batch, so the
                // chunks are independent. Pin the data + the contiguous col buffer once and
                // dispatch through OverfitParallel (zero-allocation, no TPL closure).
                var biasSpan = bias is not null ? bias.DataView.AsReadOnlySpan() : ReadOnlySpan<float>.Empty;

                unsafe
                {
                    fixed (float* inputPtr = input.DataView.AsReadOnlySpan(),
                                  weightsPtr = weights.DataView.AsReadOnlySpan(),
                                  outputPtr = outputNode.DataView.AsSpan(),
                                  colPtr = workspace.ColBuffer,
                                  biasPtr = biasSpan)
                    {
                        var ctx = new Conv2DForwardCtx
                        {
                            Input = inputPtr,
                            Weights = weightsPtr,
                            Output = outputPtr,
                            Col = colPtr,
                            Bias = biasPtr,
                            HasBias = bias is not null ? 1 : 0,
                            BatchSize = batchSize,
                            WorkerCount = workerCount,
                            ColLength = workspace.ColLength,
                            InC = inC,
                            H = h,
                            W = w,
                            K = k,
                            Padding = padding,
                            Stride = stride,
                            KSqInC = kSqInC,
                            SpatialOut = spatialOut,
                            OutC = outC,
                            InputPlaneLength = inputPlaneLength,
                            OutputPlaneLength = outputPlaneLength,
                        };

                        OverfitParallel.For(0, workerCount, &Conv2DForwardChunk, &ctx);
                    }
                }
            }
            finally
            {
                localWorkspace?.Dispose();
            }

            if (outputNode.RequiresGrad)
            {
                graph?.Record(OpCode.Conv2D, outputNode, input, weights, inC, outC, h, w, PackConvParams(k, padding, stride), c0: bias);
            }

            return outputNode;
        }

        // k / padding / stride packed into one tape int (each &lt; 1024) — Conv2D's i0..i3 hold inC/outC/h/w.
        internal static int PackConvParams(int k, int padding, int stride) => k | (padding << 10) | (stride << 20);

        internal static void UnpackConvParams(int packed, out int k, out int padding, out int stride)
        {
            k = packed & 0x3FF;
            padding = (packed >> 10) & 0x3FF;
            stride = (packed >> 20) & 0x3FF;
        }

        public static void Conv2DBackward(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode weights,
            AutogradNode output,
            AutogradNode bias,
            int inC,
            int outC,
            int h,
            int w,
            int k,
            int padding,
            int stride)
        {
            var biasNeedsGrad = bias is not null && bias.RequiresGrad;
            if (!input.RequiresGrad && !weights.RequiresGrad && !biasNeedsGrad)
            {
                return;
            }

            var batchSize = input.Shape.D0;
            var outH = (h + 2 * padding - k) / stride + 1;
            var outW = (w + 2 * padding - k) / stride + 1;
            var spatialOut = outH * outW;
            var kSqInC = k * k * inC;
            var inputPlaneLength = inC * h * w;
            var outputPlaneLength = outC * spatialOut;

            // Bias gradient: dL/db[oc] = Σ over batch & spatial positions of dL/dOutput[n, oc, :].
            if (biasNeedsGrad)
            {
                var outGrad = output.GradView.AsReadOnlySpan();
                var biasGrad = bias.GradView.AsSpan();
                for (var n = 0; n < batchSize; n++)
                {
                    for (var oc = 0; oc < outC; oc++)
                    {
                        var channel = outGrad.Slice(n * outputPlaneLength + oc * spatialOut, spatialOut);
                        var sum = 0f;
                        for (var i = 0; i < channel.Length; i++)
                        {
                            sum += channel[i];
                        }
                        biasGrad[oc] += sum;
                    }
                }

                if (!input.RequiresGrad && !weights.RequiresGrad)
                {
                    return;   // bias-only gradient done.
                }
            }

            var workspace = graph.GetConv2DWorkspace(batchSize, inC, outC, h, w, k, padding, stride);
            var workerCount = workspace.WorkerCount;

            if (weights.RequiresGrad)
            {
                workspace.ClearPartialWeightGradients();
            }

            // When input gradients aren't needed there is no grad buffer to pin; an empty
            // span pins to a harmless pointer the chunk body never dereferences (guarded
            // by InputRequiresGrad).
            var inputGradSpan = input.RequiresGrad
                ? input.GradView.AsSpan()
                : Span<float>.Empty;

            unsafe
            {
                fixed (float* inputPtr = input.DataView.AsReadOnlySpan(),
                              inputGradPtr = inputGradSpan,
                              weightsPtr = weights.DataView.AsReadOnlySpan(),
                              outputGradPtr = output.GradView.AsReadOnlySpan(),
                              colPtr = workspace.ColBuffer,
                              dColPtr = workspace.DColBuffer,
                              partialWeightGradPtr = workspace.PartialWeightGradientBuffer)
                {
                    var ctx = new Conv2DBackwardCtx
                    {
                        Input = inputPtr,
                        InputGrad = inputGradPtr,
                        Weights = weightsPtr,
                        OutputGrad = outputGradPtr,
                        Col = colPtr,
                        DCol = dColPtr,
                        PartialWeightGrad = partialWeightGradPtr,
                        BatchSize = batchSize,
                        WorkerCount = workerCount,
                        ColLength = workspace.ColLength,
                        PartialWeightGradLength = workspace.PartialWeightGradientLength,
                        InC = inC,
                        H = h,
                        W = w,
                        K = k,
                        Padding = padding,
                        Stride = stride,
                        KSqInC = kSqInC,
                        SpatialOut = spatialOut,
                        OutC = outC,
                        InputPlaneLength = inputPlaneLength,
                        OutputPlaneLength = outputPlaneLength,
                        InputRequiresGrad = input.RequiresGrad ? 1 : 0,
                        WeightsRequiresGrad = weights.RequiresGrad ? 1 : 0,
                    };

                    OverfitParallel.For(0, workerCount, &Conv2DBackwardChunk, &ctx);
                }
            }

            if (weights.RequiresGrad)
            {
                // Reduce the per-worker partial weight gradients into the real buffer.
                var weightsGrad = weights.GradView.AsSpan();

                for (var workerId = 0; workerId < workerCount; workerId++)
                {
                    var partial = workspace.GetPartialWeightGradient(workerId);
                    Simd.Add(weightsGrad, partial, weightsGrad);
                }
            }
        }

        // ── Conv2D forward — per-worker chunk dispatch ───────────────────────

        private unsafe struct Conv2DForwardCtx
        {
            public float* Input;
            public float* Weights;
            public float* Output;
            public float* Col;
            public float* Bias;
            public int HasBias;
            public int BatchSize;
            public int WorkerCount;
            public int ColLength;
            public int InC;
            public int H;
            public int W;
            public int K;
            public int Padding;
            public int Stride;
            public int KSqInC;
            public int SpatialOut;
            public int OutC;
            public int InputPlaneLength;
            public int OutputPlaneLength;
        }

        private static unsafe void Conv2DForwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<Conv2DForwardCtx>(contextPtr);

            var weights = new ReadOnlySpan<float>(ctx.Weights, ctx.OutC * ctx.KSqInC);

            for (var workerId = chunkStart; workerId < chunkEnd; workerId++)
            {
                var col = new Span<float>(ctx.Col + ((long)workerId * ctx.ColLength), ctx.ColLength);

                for (var n = workerId; n < ctx.BatchSize; n += ctx.WorkerCount)
                {
                    var inputSlice = new ReadOnlySpan<float>(
                        ctx.Input + ((long)n * ctx.InputPlaneLength),
                        ctx.InputPlaneLength);

                    var outputSlice = new Span<float>(
                        ctx.Output + ((long)n * ctx.OutputPlaneLength),
                        ctx.OutputPlaneLength);

                    Im2Col(inputSlice, ctx.InC, ctx.H, ctx.W, ctx.K, ctx.Stride, ctx.Padding, col);
                    MatMulRawSeq(weights, col, ctx.OutC, ctx.KSqInC, ctx.SpatialOut, outputSlice);

                    if (ctx.HasBias != 0)
                    {
                        for (var oc = 0; oc < ctx.OutC; oc++)
                        {
                            var b = ctx.Bias[oc];
                            var channel = outputSlice.Slice(oc * ctx.SpatialOut, ctx.SpatialOut);
                            for (var i = 0; i < channel.Length; i++)
                            {
                                channel[i] += b;
                            }
                        }
                    }
                }
            }
        }

        // ── Conv2D backward — per-worker chunk dispatch ──────────────────────

        private unsafe struct Conv2DBackwardCtx
        {
            public float* Input;
            public float* InputGrad;
            public float* Weights;
            public float* OutputGrad;
            public float* Col;
            public float* DCol;
            public float* PartialWeightGrad;
            public int BatchSize;
            public int WorkerCount;
            public int ColLength;
            public int PartialWeightGradLength;
            public int InC;
            public int H;
            public int W;
            public int K;
            public int Padding;
            public int Stride;
            public int KSqInC;
            public int SpatialOut;
            public int OutC;
            public int InputPlaneLength;
            public int OutputPlaneLength;
            public int InputRequiresGrad;
            public int WeightsRequiresGrad;
        }

        private static unsafe void Conv2DBackwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<Conv2DBackwardCtx>(contextPtr);

            var weights = new ReadOnlySpan<float>(ctx.Weights, ctx.OutC * ctx.KSqInC);

            for (var workerId = chunkStart; workerId < chunkEnd; workerId++)
            {
                var col = new Span<float>(ctx.Col + ((long)workerId * ctx.ColLength), ctx.ColLength);
                var dCol = new Span<float>(ctx.DCol + ((long)workerId * ctx.ColLength), ctx.ColLength);
                var partialWeightGrad = new Span<float>(
                    ctx.PartialWeightGrad + ((long)workerId * ctx.PartialWeightGradLength),
                    ctx.PartialWeightGradLength);

                for (var n = workerId; n < ctx.BatchSize; n += ctx.WorkerCount)
                {
                    var inputSlice = new ReadOnlySpan<float>(
                        ctx.Input + ((long)n * ctx.InputPlaneLength),
                        ctx.InputPlaneLength);

                    Im2Col(inputSlice, ctx.InC, ctx.H, ctx.W, ctx.K, ctx.Stride, ctx.Padding, col);

                    var outGradSlice = new ReadOnlySpan<float>(
                        ctx.OutputGrad + ((long)n * ctx.OutputPlaneLength),
                        ctx.OutputPlaneLength);

                    if (ctx.WeightsRequiresGrad != 0)
                    {
                        MatMulAdd_A_BT_Seq(
                            outGradSlice,
                            col,
                            partialWeightGrad,
                            ctx.OutC,
                            ctx.SpatialOut,
                            ctx.KSqInC);
                    }

                    if (ctx.InputRequiresGrad != 0)
                    {
                        dCol.Clear();

                        MatMulAdd_AT_B_Seq(
                            weights,
                            outGradSlice,
                            dCol,
                            ctx.OutC,
                            ctx.KSqInC,
                            ctx.SpatialOut);

                        var inputGradSlice = new Span<float>(
                            ctx.InputGrad + ((long)n * ctx.InputPlaneLength),
                            ctx.InputPlaneLength);

                        Col2Im(dCol, ctx.InC, ctx.H, ctx.W, ctx.K, ctx.Stride, ctx.Padding, inputGradSlice);
                    }
                }
            }
        }

        // ====================================================================
        // IM2COL / COL2IM
        // ====================================================================

        public static void Im2Col(
            ReadOnlySpan<float> input,
            int channels,
            int h,
            int w,
            int k,
            int s,
            int p,
            Span<float> output)
        {
            var oH = (h + 2 * p - k) / s + 1;
            var oW = (w + 2 * p - k) / s + 1;

            for (var c = 0; c < channels; c++)
            {
                for (var kh = 0; kh < k; kh++)
                {
                    for (var kw = 0; kw < k; kw++)
                    {
                        var rowOffset = (c * k * k + kh * k + kw) * oH * oW;

                        for (var y = 0; y < oH; y++)
                        {
                            var inputY = y * s - p + kh;

                            if (inputY >= 0 && inputY < h)
                            {
                                var inputOffset = c * h * w + inputY * w;

                                for (var x = 0; x < oW; x++)
                                {
                                    var inputX = x * s - p + kw;
                                    output[rowOffset + y * oW + x] =
                                        inputX >= 0 && inputX < w
                                            ? input[inputOffset + inputX]
                                            : 0f;
                                }
                            }
                            else
                            {
                                output
                                    .Slice(rowOffset + y * oW, oW)
                                    .Clear();
                            }
                        }
                    }
                }
            }
        }

        public static void Col2Im(
            ReadOnlySpan<float> col,
            int channels,
            int h,
            int w,
            int k,
            int s,
            int p,
            Span<float> gradientInput)
        {
            var oH = (h + 2 * p - k) / s + 1;
            var oW = (w + 2 * p - k) / s + 1;

            for (var c = 0; c < channels; c++)
            {
                for (var kh = 0; kh < k; kh++)
                {
                    for (var kw = 0; kw < k; kw++)
                    {
                        var rowOffset = (c * k * k + kh * k + kw) * oH * oW;

                        for (var y = 0; y < oH; y++)
                        {
                            var inputY = y * s - p + kh;

                            if (inputY >= 0 && inputY < h)
                            {
                                var inputOffset = c * h * w + inputY * w;

                                for (var x = 0; x < oW; x++)
                                {
                                    var inputX = x * s - p + kw;

                                    if (inputX >= 0 && inputX < w)
                                    {
                                        gradientInput[inputOffset + inputX] += col[rowOffset + y * oW + x];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
