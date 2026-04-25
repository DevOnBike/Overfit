// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Runtime;

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
            int k)
        {
            var outH = h - k + 1;
            var outW = w - k + 1;
            var batchSize = input.Shape.D0;
            var kSqInC = k * k * inC;
            var spatialOut = outH * outW;
            var inputPlaneLength = inC * h * w;
            var outputPlaneLength = outC * spatialOut;

            var outputNode = AllocateNode(
                graph,
                new TensorShape(batchSize, outC, outH, outW),
                input.RequiresGrad || weights.RequiresGrad,
                clearMemory: true);

            // The convolution workspace holds per-worker im2col scratch buffers that the
            // Parallel.For below writes into. Normally it lives on the graph and is reused
            // across all conv ops in a training step (zero per-call allocation). For
            // inference paths that pass graph=null (e.g. a lightweight forward over a
            // single image without recording a tape), we create a local workspace for the
            // duration of the call. The local path allocates but isn't on the training
            // hot path; it's the cost of decoupling inference from graph state.
            Conv2DWorkspace? localWorkspace = null;
            Conv2DWorkspace workspace;

            if (graph is not null)
            {
                workspace = graph.GetConv2DWorkspace(batchSize, inC, outC, h, w, k);
            }
            else
            {
                localWorkspace = new Conv2DWorkspace();
                var workerCount = Math.Max(1, Math.Min(OverfitParallel.MaxDegreeOfParallelism, Math.Max(1, batchSize)));
                localWorkspace.Ensure(
                    workerCount,
                    colLength: kSqInC * spatialOut,
                    partialWeightGradientLength: outC * kSqInC);
                workspace = localWorkspace;
            }

            try
            {
                var workerCount = workspace.WorkerCount;

                Parallel.For(0, workerCount, OverfitParallel.Options, workerId =>
                {
                    var col = workspace.GetColBuffer(workerId);

                    for (var n = workerId; n < batchSize; n += workerCount)
                    {
                        var inputSlice = input
                            .DataView
                            .AsReadOnlySpan()
                            .Slice(n * inputPlaneLength, inputPlaneLength);

                        var weightsSpan = weights
                            .DataView
                            .AsReadOnlySpan();

                        var outputSlice = outputNode
                            .DataView
                            .AsSpan()
                            .Slice(n * outputPlaneLength, outputPlaneLength);

                        Im2Col(
                            inputSlice,
                            inC,
                            h,
                            w,
                            k,
                            1,
                            0,
                            col);

                        MatMulRawSeq(
                            weightsSpan,
                            col,
                            outC,
                            kSqInC,
                            spatialOut,
                            outputSlice);
                    }
                });
            }
            finally
            {
                localWorkspace?.Dispose();
            }

            if (outputNode.RequiresGrad)
            {
                graph?.Record(OpCode.Conv2D, outputNode, input, weights, inC, outC, h, w, k);
            }

            return outputNode;
        }

        public static void Conv2DBackward(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode weights,
            AutogradNode output,
            int inC,
            int outC,
            int h,
            int w,
            int k)
        {
            if (!input.RequiresGrad && !weights.RequiresGrad)
            {
                return;
            }

            var batchSize = input.Shape.D0;
            var outH = h - k + 1;
            var outW = w - k + 1;
            var spatialOut = outH * outW;
            var kSqInC = k * k * inC;
            var inputPlaneLength = inC * h * w;
            var outputPlaneLength = outC * spatialOut;
            var partialWeightGradientLength = outC * kSqInC;

            var workspace = graph.GetConv2DWorkspace(batchSize, inC, outC, h, w, k);
            var workerCount = workspace.WorkerCount;

            if (weights.RequiresGrad)
            {
                workspace.ClearPartialWeightGradients();
            }

            Parallel.For(0, workerCount, OverfitParallel.Options, workerId =>
            {
                var col = workspace.GetColBuffer(workerId);
                var dCol = workspace.GetDColBuffer(workerId);
                var partialWeightGrad = workspace.GetPartialWeightGradient(workerId);

                for (var n = workerId; n < batchSize; n += workerCount)
                {
                    var inputSlice = input
                        .DataView
                        .AsReadOnlySpan()
                        .Slice(n * inputPlaneLength, inputPlaneLength);

                    Im2Col(
                        inputSlice,
                        inC,
                        h,
                        w,
                        k,
                        1,
                        0,
                        col);

                    var outGradSlice = output
                        .GradView
                        .AsReadOnlySpan()
                        .Slice(n * outputPlaneLength, outputPlaneLength);

                    if (weights.RequiresGrad)
                    {
                        MatMulAdd_A_BT_Seq(
                            outGradSlice,
                            col,
                            partialWeightGrad,
                            outC,
                            spatialOut,
                            kSqInC);
                    }

                    if (input.RequiresGrad)
                    {
                        dCol.Clear();

                        var weightsSpan = weights
                            .DataView
                            .AsReadOnlySpan();

                        MatMulAdd_AT_B_Seq(
                            weightsSpan,
                            outGradSlice,
                            dCol,
                            outC,
                            kSqInC,
                            spatialOut);

                        var inputGradSlice = input
                            .GradView
                            .AsSpan()
                            .Slice(n * inputPlaneLength, inputPlaneLength);

                        Col2Im(
                            dCol,
                            inC,
                            h,
                            w,
                            k,
                            1,
                            0,
                            inputGradSlice);
                    }
                }
            });

            if (weights.RequiresGrad)
            {
                var weightsGrad = weights.GradView.AsSpan();

                for (var workerId = 0; workerId < workerCount; workerId++)
                {
                    var partialWeightGrad = workspace
                        .GetPartialWeightGradient(workerId)
                        .Slice(0, partialWeightGradientLength);

                    Simd.Add(weightsGrad, partialWeightGrad, weightsGrad);
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