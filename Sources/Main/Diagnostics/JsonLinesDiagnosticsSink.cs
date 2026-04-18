// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Data.Serialization;
using DevOnBike.Overfit.Diagnostics.Contracts;

namespace DevOnBike.Overfit.Diagnostics
{
    public sealed class JsonLinesDiagnosticsSink : IOverfitDiagnosticsSink, IDisposable
    {
        private readonly TextWriter _writer;
        private readonly Lock _gate = new();
        private readonly bool _ownsWriter;
        private volatile bool _disposed;

        public JsonLinesDiagnosticsSink(TextWriter writer, bool ownsWriter = false)
        {
            _writer = writer ?? throw new ArgumentNullException(nameof(writer));
            _ownsWriter = ownsWriter;
        }

        public static JsonLinesDiagnosticsSink CreateFile(string path, bool append = false)
        {
            var stream = new FileStream(path, append ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.Read);
            var writer = new StreamWriter(stream) { AutoFlush = true };

            return new JsonLinesDiagnosticsSink(writer, ownsWriter: true);
        }

        public void OnKernelCompleted(in KernelDiagnosticEvent evt)
        {
            WriteEnvelope("kernel", evt);
        }

        public void OnModuleCompleted(in ModuleDiagnosticEvent evt)
        {
            WriteEnvelope("module", evt);
        }

        public void OnGraphCompleted(in GraphDiagnosticEvent evt)
        {
            WriteEnvelope("graph", evt);
        }

        public void OnAllocation(in AllocationDiagnosticEvent evt)
        {
            WriteEnvelope("allocation", evt);
        }

        public void OnCounter(string name, long value)
        {
            WriteEnvelope("counter", new
            {
                name,
                value
            });
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed)
                {
                    return;
                }

                _disposed = true;

                if (_ownsWriter)
                {
                    _writer.Dispose();
                }
            }
        }

        private void WriteEnvelope<T>(string type, T payload)
        {
            if (_disposed)
            {
                return;
            }

            lock (_gate)
            {
                if (_disposed)
                {
                    return;
                }

                var envelope = new { tsUtc = DateTime.UtcNow, type, payload };
                _writer.WriteLine(JsonSerializer.Serialize(envelope));
            }
        }
    }
}