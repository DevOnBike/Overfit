// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // MAX POOL 2D
        // ====================================================================

        public static AutogradNode MaxPool2D(ComputationGraph graph, AutogradNode input, int channels, int h, int w, int pool)
        {
            var batchSize = input.DataView.GetDim(0);
            int oH = h / pool, oW = w / pool;

            var outD = new FastTensor<float>(batchSize, channels, oH, oW, false);
            var maxIndices = new FastTensor<float>(batchSize, channels, oH, oW, false);

            if (batchSize < BatchSequentialThreshold)
            {
                ref var iR = ref MemoryMarshal.GetReference(input.DataView.AsSpan());
                ref var oR = ref MemoryMarshal.GetReference(outD.GetView().AsSpan());
                ref var xR = ref MemoryMarshal.GetReference(maxIndices.GetView().AsSpan());

                for (var n = 0; n < batchSize; n++)
                {
                    ExecuteMaxPoolSingleBatch(n, channels, h, w, oH, oW, pool, ref iR, ref oR, ref xR);
                }
            }
            else
            {
                Parallel.For(0, batchSize, n =>
                {
                    ref var iR = ref MemoryMarshal.GetReference(input.DataView.AsSpan());
                    ref var oR = ref MemoryMarshal.GetReference(outD.GetView().AsSpan());
                    ref var xR = ref MemoryMarshal.GetReference(maxIndices.GetView().AsSpan());

                    ExecuteMaxPoolSingleBatch(n, channels, h, w, oH, oW, pool, ref iR, ref oR, ref xR);
                });
            }

            var output = new AutogradNode(outD, input.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.MaxPool2D, output, input, new AutogradNode(maxIndices));
            }
            else
            {
                maxIndices.Dispose();
            }

            return output;
        }

        private static void ExecuteMaxPoolSingleBatch(int n, int channels, int h, int w, int oH, int oW, int pool, ref float iR, ref float oR, ref float xR)
        {
            var batchOffset = n * channels * h * w;
            var outBatchOffset = n * channels * oH * oW;

            for (var c = 0; c < channels; c++)
            {
                var channelOffset = batchOffset + c * h * w;
                var outChannelOffset = outBatchOffset + c * oH * oW;

                for (var oh = 0; oh < oH; oh++)
                {
                    for (var ow = 0; ow < oW; ow++)
                    {
                        var maxVal = float.MinValue;
                        var maxIdx = 0;

                        for (var ph = 0; ph < pool; ph++)
                        {
                            for (var pw = 0; pw < pool; pw++)
                            {
                                var ih = oh * pool + ph;
                                var iw = ow * pool + pw;
                                var idx = channelOffset + ih * w + iw;
                                var val = Unsafe.Add(ref iR, idx);

                                if (val > maxVal)
                                {
                                    maxVal = val;
                                    maxIdx = idx;
                                }
                            }
                        }

                        var outIdx = outChannelOffset + oh * oW + ow;
                        Unsafe.Add(ref oR, outIdx) = maxVal;
                        Unsafe.Add(ref xR, outIdx) = maxIdx;
                    }
                }
            }
        }

        public static void MaxPool2DBackward(AutogradNode input, AutogradNode maxIndices, AutogradNode output)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var oGS = output.GradView.AsReadOnlySpan();
            var iGS = input.GradView.AsSpan();
            var idxS = maxIndices.DataView.AsReadOnlySpan();

            for (var i = 0; i < oGS.Length; i++)
            {
                var maxIdx = (int)idxS[i];
                iGS[maxIdx] += oGS[i];
            }
        }

        // ====================================================================
        // GLOBAL AVERAGE POOL 2D
        // ====================================================================

        public static AutogradNode GlobalAveragePool2D(ComputationGraph graph, AutogradNode input, int channels, int h, int w)
        {
            var batchSize = input.DataView.GetDim(0);
            var spatialSize = h * w;
            var scale = 1f / spatialSize;

            var outD = new FastTensor<float>(batchSize, channels, false);
            var inS = input.DataView.AsReadOnlySpan();
            var outS = outD.GetView().AsSpan();

            if (batchSize < BatchSequentialThreshold)
            {
                for (var n = 0; n < batchSize; n++)
                {
                    for (var c = 0; c < channels; c++)
                    {
                        var channelSlice = inS.Slice(n * channels * spatialSize + c * spatialSize, spatialSize);
                        outS[n * channels + c] = TensorPrimitives.Sum(channelSlice) * scale;
                    }
                }
            }
            else
            {
                Parallel.For(0, batchSize, n =>
                {
                    for (var c = 0; c < channels; c++)
                    {
                        var channelSlice = input.DataView.AsReadOnlySpan().Slice(n * channels * spatialSize + c * spatialSize, spatialSize);
                        outD.GetView().AsSpan()[n * channels + c] = TensorPrimitives.Sum(channelSlice) * scale;
                    }
                });
            }

            var output = new AutogradNode(outD, input.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.GlobalAveragePool2D, output, input, null, h, w, channels);
            }

            return output;
        }

        public static void GlobalAvgPool2DBackward(AutogradNode input, AutogradNode output, int h, int w, int channels)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var batchSize = input.DataView.GetDim(0);
            var spatialSize = h * w;
            var scale = 1f / spatialSize;

            var oGS = output.GradView.AsReadOnlySpan();
            var iGS = input.GradView.AsSpan();

            for (var n = 0; n < batchSize; n++)
            {
                for (var c = 0; c < channels; c++)
                {
                    var grad = oGS[n * channels + c] * scale;
                    var channelSlice = iGS.Slice(n * channels * spatialSize + c * spatialSize, spatialSize);

                    for (var i = 0; i < spatialSize; i++)
                    {
                        channelSlice[i] += grad;
                    }
                }
            }
        }
    }
}
