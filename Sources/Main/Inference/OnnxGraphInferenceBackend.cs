// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Inference.Contracts;

namespace DevOnBike.Overfit.Inference
{
    /// <summary>
    /// <see cref="IInferenceBackend"/> implementation for ONNX DAG models.
    /// Wraps <see cref="Onnx.OnnxGraphModel"/> and exposes it through
    /// <see cref="InferenceEngine.FromBackend"/>.
    /// </summary>
    public sealed class OnnxGraphInferenceBackend : IInferenceBackend
    {
        private readonly Onnx.OnnxGraphModel _model;

        public OnnxGraphInferenceBackend(Onnx.OnnxGraphModel model)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
        }

        public int InputSize  => _model.InputSize;
        public int OutputSize => _model.OutputSize;

        public void Run(ReadOnlySpan<float> input, Span<float> output)
            => _model.RunInference(input, output);

        public void Warmup(int iterations)
        {
            var dummyInput  = new float[InputSize];
            var dummyOutput = new float[OutputSize];

            for (var i = 0; i < iterations; i++)
            {
                _model.RunInference(dummyInput, dummyOutput);
            }
        }

        public void Dispose() => _model.Dispose();
    }
}
