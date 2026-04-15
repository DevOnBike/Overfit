// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics.Contracts;

namespace DevOnBike.Overfit.Diagnostics
{
    public sealed class TextWriterDiagnosticsSink : IOverfitDiagnosticsSink, IDisposable
    {
        private readonly TextWriter _writer;
        private readonly Lock _gate = new();
        private readonly bool _ownsWriter;

        public TextWriterDiagnosticsSink(TextWriter writer, bool ownsWriter = false)
        {
            _writer = writer ?? throw new ArgumentNullException(nameof(writer));
            _ownsWriter = ownsWriter;
        }

        public static TextWriterDiagnosticsSink CreateFile(string path, bool append = false)
        {
            var stream = new FileStream(path, append ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.Read);
            var writer = new StreamWriter(stream) { AutoFlush = true };
            return new TextWriterDiagnosticsSink(writer, ownsWriter: true);
        }

        public void OnKernelCompleted(in KernelDiagnosticEvent evt)
        {
            WriteLine(
            FormattableString.Invariant(
            $"KERNEL category={evt.Category} name={evt.Name} phase={evt.Phase} train={evt.IsTraining} batch={evt.BatchSize} features={evt.FeatureCount} in={evt.InputElements} out={evt.OutputElements} ms={evt.DurationMs:F4}"));
        }

        public void OnModuleCompleted(in ModuleDiagnosticEvent evt)
        {
            WriteLine(
            FormattableString.Invariant(
            $"MODULE type={evt.ModuleType} phase={evt.Phase} train={evt.IsTraining} batch={evt.BatchSize} inRows={evt.InputRows} inCols={evt.InputCols} outRows={evt.OutputRows} outCols={evt.OutputCols} alloc={evt.AllocatedBytes} ms={evt.DurationMs:F4}"));
        }

        public void OnGraphCompleted(in GraphDiagnosticEvent evt)
        {
            WriteLine(
            FormattableString.Invariant(
            $"GRAPH phase={evt.Phase} train={evt.IsTraining} batch={evt.BatchSize} tapeOps={evt.TapeOpCount} alloc={evt.AllocatedBytes} backwardMs={evt.BackwardMs:F4} gc0={evt.Gen0Collections} gc1={evt.Gen1Collections} gc2={evt.Gen2Collections}"));
        }

        public void OnAllocation(in AllocationDiagnosticEvent evt)
        {
            WriteLine(
            FormattableString.Invariant(
            $"ALLOC owner={evt.Owner} type={evt.ResourceType} elements={evt.Elements} bytes={evt.Bytes} pooled={evt.IsPooled} managed={evt.IsManaged}"));
        }

        public void OnCounter(string name, long value)
        {
            WriteLine(FormattableString.Invariant($"COUNTER name={name} delta={value}"));
        }

        public void Dispose()
        {
            if (_ownsWriter)
            {
                _writer.Dispose();
            }
        }

        private void WriteLine(string line)
        {
            lock (_gate)
            {
                _writer.WriteLine(line);
            }
        }
    }
}
