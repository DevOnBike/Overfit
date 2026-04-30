// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.Autograd
{
    public sealed partial class ComputationGraph
    {
        /// <summary>
        /// 2-D convolution (NCHW layout, VALID padding — no border padding).
        /// Output shape: [batch, outChannels, inputH - kernelSize + 1, inputW - kernelSize + 1].
        /// </summary>
        public AutogradNode Conv2D(
            AutogradNode input,
            AutogradNode weights,
            int inChannels,
            int outChannels,
            int inputH,
            int inputW,
            int kernelSize)
        {
            return TensorMath.Conv2D(
                this,
                input,
                weights,
                inChannels,
                outChannels,
                inputH,
                inputW,
                kernelSize);
        }

        /// <summary>
        /// 2-D max pooling (NCHW layout).
        /// Output shape: [batch, channels, inputH / pool, inputW / pool].
        /// </summary>
        public AutogradNode MaxPool2D(
            AutogradNode input,
            int channels,
            int inputH,
            int inputW,
            int pool)
        {
            return TensorMath.MaxPool2D(this, input, channels, inputH, inputW, pool);
        }

        /// <summary>
        /// Global average pooling over spatial dimensions.
        /// Output shape: [batch, channels].
        /// </summary>
        public AutogradNode GlobalAveragePool2D(
            AutogradNode input,
            int channels,
            int inputH,
            int inputW)
        {
            return TensorMath.GlobalAveragePool2D(this, input, channels, inputH, inputW);
        }
    }
}
