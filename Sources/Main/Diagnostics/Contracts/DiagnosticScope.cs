// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;

namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly ref struct DiagnosticScope
    {
        private readonly string _category;
        private readonly string _name;
        private readonly string _phase;
        private readonly bool _isTraining;
        private readonly int _batchSize;
        private readonly int _featureCount;
        private readonly int _inputElements;
        private readonly int _outputElements;
        private readonly long _start;

        public DiagnosticScope(
            string category,
            string name,
            string phase = "forward",
            bool isTraining = false,
            int batchSize = 0,
            int featureCount = 0,
            int inputElements = 0,
            int outputElements = 0)
        {
            _category = category;
            _name = name;
            _phase = phase;
            _isTraining = isTraining;
            _batchSize = batchSize;
            _featureCount = featureCount;
            _inputElements = inputElements;
            _outputElements = outputElements;
            _start = Stopwatch.GetTimestamp();
        }

        public void Dispose()
        {
            if (!OverfitDiagnostics.IsEnabled())
            {
                return;
            }

            var end = Stopwatch.GetTimestamp();
            var ms = (end - _start) * 1000.0 / Stopwatch.Frequency;

            OverfitDiagnostics.KernelCompleted(new KernelDiagnosticEvent(
            Category: _category,
            Name: _name,
            DurationMs: ms,
            Phase: _phase,
            IsTraining: _isTraining,
            BatchSize: _batchSize,
            FeatureCount: _featureCount,
            InputElements: _inputElements,
            OutputElements: _outputElements));
        }
    }
}
