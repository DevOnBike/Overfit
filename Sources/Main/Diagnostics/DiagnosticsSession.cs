namespace DevOnBike.Overfit.Diagnostics
{
    /// <summary>
    /// Temporary diagnostics session that safely swaps the global diagnostics settings
    /// and restores the previous state on dispose.
    /// </summary>
    public sealed class DiagnosticsSession : IDisposable
    {
        private readonly bool _oldEnabled;
        private readonly IOverfitDiagnosticsSink _oldSink;
        private bool _disposed;

        public DiagnosticsSession(bool enabled, IOverfitDiagnosticsSink sink)
        {
            _oldEnabled = OverfitDiagnostics.Enabled;
            _oldSink = OverfitDiagnostics.Sink;

            OverfitDiagnostics.Enabled = enabled;
            OverfitDiagnostics.Sink = sink ?? NullOverfitDiagnosticsSink.Instance;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            OverfitDiagnostics.Enabled = _oldEnabled;
            OverfitDiagnostics.Sink = _oldSink;
            
            _disposed = true;
        }
    }
}
