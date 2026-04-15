// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics.Contracts;

namespace DevOnBike.Overfit.Diagnostics
{
    public sealed class CompositeOverfitDiagnosticsSink : IOverfitDiagnosticsSink
    {
        private readonly IOverfitDiagnosticsSink[] _sinks;

        public CompositeOverfitDiagnosticsSink(params IOverfitDiagnosticsSink[] sinks)
        {
            if (sinks == null || sinks.Length == 0)
            {
                _sinks = [];
                return;
            }

            var count = 0;
            for (var i = 0; i < sinks.Length; i++)
            {
                if (sinks[i] != null)
                {
                    count++;
                }
            }

            if (count == 0)
            {
                _sinks = [];
                return;
            }

            _sinks = new IOverfitDiagnosticsSink[count];

            var dst = 0;
            for (var i = 0; i < sinks.Length; i++)
            {
                var sink = sinks[i];
                if (sink != null)
                {
                    _sinks[dst++] = sink;
                }
            }
        }

        public void OnKernelCompleted(in KernelDiagnosticEvent evt)
        {
            for (var i = 0; i < _sinks.Length; i++)
            {
                _sinks[i].OnKernelCompleted(evt);
            }
        }

        public void OnModuleCompleted(in ModuleDiagnosticEvent evt)
        {
            for (var i = 0; i < _sinks.Length; i++)
            {
                _sinks[i].OnModuleCompleted(evt);
            }
        }

        public void OnGraphCompleted(in GraphDiagnosticEvent evt)
        {
            for (var i = 0; i < _sinks.Length; i++)
            {
                _sinks[i].OnGraphCompleted(evt);
            }
        }

        public void OnAllocation(in AllocationDiagnosticEvent evt)
        {
            for (var i = 0; i < _sinks.Length; i++)
            {
                _sinks[i].OnAllocation(evt);
            }
        }

        public void OnCounter(string name, long value)
        {
            for (var i = 0; i < _sinks.Length; i++)
            {
                _sinks[i].OnCounter(name, value);
            }
        }
    }
}