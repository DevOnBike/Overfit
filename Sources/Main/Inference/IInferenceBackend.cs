// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Inference
{
    public interface IInferenceBackend : IDisposable
    {
        int InputSize { get; }

        int OutputSize { get; }

        void Run(
            ReadOnlySpan<float> input,
            Span<float> output);

        void Warmup(int iterations);
    }
}
