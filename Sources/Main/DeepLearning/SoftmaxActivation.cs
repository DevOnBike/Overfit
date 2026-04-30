// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Standalone softmax activation applied per-row (axis=-1).
    /// Output sums to 1 per sample. Used as the final layer in multi-class classifiers
    /// when the model is exported without fused cross-entropy loss.
    ///
    /// Note: during training, prefer <see cref="TensorMath.SoftmaxCrossEntropy"/> which
    /// fuses softmax and loss for numerical stability. This module is provided for
    /// ONNX import compatibility (PyTorch exports a standalone Softmax node when
    /// using <c>nn.Softmax(dim=-1)</c> as the final layer).
    /// </summary>
    public sealed class SoftmaxActivation : IModule
    {
        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;

        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            var batchSize  = input.Shape.D0;
            var outputSize = input.Shape.D1;

            var output = TensorMath.AllocateNode(
                graph,
                input.Shape,
                input.RequiresGrad,
                clearMemory: false);

            var inS  = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();

            for (var b = 0; b < batchSize; b++)
            {
                TensorMath.StableSoftmax(
                    inS.Slice(b * outputSize, outputSize),
                    outS.Slice(b * outputSize, outputSize));
            }

            return output;
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            // Inference: single sample (size = outputSize).
            TensorMath.StableSoftmax(input, output);
        }

        public IEnumerable<AutogradNode> Parameters() => [];

        public void InvalidateParameterCaches() { }

        public void Save(BinaryWriter bw) { }

        public void Load(BinaryReader br) { }

        public void Dispose() { }
    }
}
