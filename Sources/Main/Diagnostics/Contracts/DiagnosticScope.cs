// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;

namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly ref struct DiagnosticScope
    {
        private readonly string _name;
        private readonly string _category;
        private readonly long _start;
        private readonly int _batch;
        private readonly int _features;

        public DiagnosticScope(string category, string name, int batch = 0, int features = 0)
        {
            _category = category;
            _name = name;
            _batch = batch;
            _features = features;
            _start = Stopwatch.GetTimestamp();
        }

        public void Dispose()
        {
            if (!OverfitDiagnostics.IsEnabled())
            {
                return;
            }

            long end = Stopwatch.GetTimestamp();
            double ms = (end - _start) * 1000.0 / Stopwatch.Frequency;

            OverfitDiagnostics.Sink.OnKernelCompleted(new KernelDiagnosticEvent(
            _category,
            _name,
            ms,
            _batch,
            _features));
        }
    }
}