// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using ILGPU.Runtime;
using Spectre.Console;
using Spectre.Console.Rendering;

// Spectre.Console.Text collides with the System.Text NAMESPACE that ImplicitUsings brings in, so the
// bare name resolves to the namespace and fails. Aliased rather than fully qualified at each use.
using SpectreText = Spectre.Console.Text;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The live console view: which card, how much of its memory is in use, how hot it is, how busy it
    /// is, and the throughput of the last completed measurement.
    /// <para>
    /// <b>Two structural decisions, both of which exist to protect the measurement rather than to look
    /// tidy.</b>
    /// </para>
    /// <para>
    /// <b>1. Refresh is PULLED by the probe, never pushed by a timer.</b> Spectre's <c>Live</c> display
    /// can repaint on its own schedule; this one does not. <see cref="Refresh"/> is called by the probe
    /// at points it knows are outside every timed region, so a repaint can never land between a
    /// timestamp and its synchronise. A background repaint thread would burn host cycles concurrently
    /// with C1, C2 and C3, which run on the host - the perturbation would be real, variable and
    /// invisible. Pulling makes the guarantee structural instead of hopeful, and the residual is
    /// measured by <c>--live-perturbation</c> rather than assumed away.
    /// </para>
    /// <para>
    /// <b>2. It renders to STANDARD ERROR.</b> The report goes to standard output and the README tells a
    /// stranger to copy everything between two marker lines. A live display interleaved into that stream
    /// would shred the one artefact this whole exercise exists to produce.
    /// </para>
    /// </summary>
    internal sealed class LiveView : IDisposable
    {
        private readonly IGpuTelemetry _telemetry;
        private readonly LiveProbeState _state;
        private readonly IAnsiConsole _console;

        private static bool _countersReset;

        private LiveDisplayContext? _context;
        private TelemetrySample? _latest;

        private LiveView(IGpuTelemetry telemetry, LiveProbeState state, IAnsiConsole console)
        {
            _telemetry = telemetry;
            _state = state;
            _console = console;
        }

        /// <summary>Why the live view is not running, or null when it is.</summary>
        public static string? Unavailable { get; private set; }

        /// <summary>
        /// While true, <see cref="Refresh"/> does nothing. Set around a timed phase when the measured
        /// perturbation says the repaint is material.
        /// </summary>
        public bool Suspended { get; set; }

        /// <summary>
        /// Frames actually painted during the run, and repaints that were requested from INSIDE a timed
        /// region and dropped. Both go into the report, and the pair is the point: a stranger's artefact
        /// then says that the observer was running AND that it never touched a clock. Either number alone
        /// is ambiguous - zero frames and zero drops is also what a view that never started looks like.
        /// <para>
        /// Static because the report is rendered after the view is disposed, and because one drop anywhere
        /// in the run is enough to make the cadence wrong for the whole of it.
        /// </para>
        /// </summary>
        public static int FramesPainted { get; private set; }

        /// <inheritdoc cref="FramesPainted"/>
        public static int DroppedInsideTimedRegion { get; private set; }

        /// <summary>
        /// Puts both counters back to zero. <b>For <see cref="GuardSelfCheck"/> only, and only before
        /// anything is measured.</b>
        /// <para>
        /// The self-check has to drive <see cref="Refresh"/> from inside a real timed region to prove the
        /// guard fires, which moves the drop counter; leaving that increment behind would make every run
        /// report a drop it did not suffer, and the report treats a non-zero drop count as a defect
        /// worth escalating. Called a second time it would do the opposite and erase a real one, so it
        /// refuses: the counters are the only evidence that the repaints stayed out of the clocks.
        /// </para>
        /// </summary>
        public static void ResetCounters()
        {
            if (_countersReset)
            {
                throw new InvalidOperationException(
                    "LiveView.ResetCounters may be called once, by the self-check, before anything is " +
                    "measured. A later call would erase a real drop count, which is the evidence that no " +
                    "repaint landed inside a timed region.");
            }

            _countersReset = true;
            FramesPainted = 0;
            DroppedInsideTimedRegion = 0;
        }

        /// <summary>What the view shows. The probe writes to it; the view only reads it.</summary>
        public LiveProbeState State => _state;

        /// <summary>
        /// Returns null and sets <see cref="Unavailable"/> rather than throwing. A redirected error
        /// stream is the common case - it would fill a log with repaint frames instead of showing a
        /// display to anybody.
        /// </summary>
        /// <param name="force">
        /// Build the view even when standard error is redirected. Only <c>--live-perturbation</c> passes
        /// this, and it does so because the cost being measured is the LAYOUT AND WRITE of a frame, which
        /// happens either way - and because a harness that cannot run under a captured stream cannot be
        /// run by anything that records its result. A run forced this way says so in the report.
        /// </param>
        public static LiveView? TryCreate(IGpuTelemetry telemetry, LiveProbeState state, bool force = false)
        {
            if (Console.IsErrorRedirected && !force)
            {
                Unavailable = "standard error is redirected, so there is no terminal to draw on";
                return null;
            }

            try
            {
                var console = AnsiConsole.Create(new AnsiConsoleSettings
                {
                    Out = new AnsiConsoleOutput(Console.Error),
                });

                return new LiveView(telemetry, state, console);
            }
            catch (Exception ex) when (ex is IOException or InvalidOperationException or NotSupportedException)
            {
                Unavailable = $"the console could not be prepared ({ex.Message.Trim()})";
                return null;
            }
        }

        /// <summary>
        /// The whole decision for a real run: whether a view was asked for, what may drive it, and what
        /// the report should say when there is none. Returns null in every case that is not a working
        /// view, and the run then proceeds without one - the established shape of
        /// <see cref="CuBlasArm.TryCreate"/>, and the reason <c>--live</c> on a machine with no NVIDIA
        /// driver prints a reason instead of throwing.
        /// <para>
        /// <b>A missing driver does NOT silently fall back to the stub.</b> The stub's readings are
        /// synthetic, and a synthetic temperature shown under the heading of a real card is a fabricated
        /// reading in the exact sense this probe refuses everywhere else. <c>--live-stub</c> is the
        /// explicit way to ask for it, and the report names the source either way.
        /// </para>
        /// <para>
        /// <b>Every one of the three refusals says so on the CONSOLE, at the moment it is taken, as well
        /// as in the report.</b> The report is the durable record and the console line is the immediate
        /// one, and only the second is any use to somebody watching a twelve-minute run start: a report
        /// note is read when the run is over. The case with no message at all was the worst of the three,
        /// because a user who forgot <c>--live</c> got total silence and could not tell it apart from a
        /// broken view. <see cref="Announce"/> writes both from one place so the two cannot diverge.
        /// </para>
        /// </summary>
        /// <param name="log">
        /// Standard error, which is also what the view draws on. Written to before the display starts,
        /// never during it - Spectre owns the stream for the whole of <see cref="Run"/>.
        /// </param>
        public static LiveView? Open(
            ProbeOptions options,
            Accelerator accelerator,
            LiveProbeState state,
            Report report,
            TextWriter log)
        {
            if (!options.Live)
            {
                Announce(
                    report,
                    log,
                    "NOT ASKED FOR - --live was not passed, so no live view ran. Nothing else about this run " +
                    "depended on one. Pass --live to watch the card while the probe runs, or --live-stub to " +
                    "see the panels driven by synthetic readings on a machine with no NVIDIA driver.");
                return null;
            }

            var telemetry = options.LiveStub
                ? new StubTelemetry(options.Seed)
                : (IGpuTelemetry?)NvmlTelemetry.TryCreate(accelerator);

            if (telemetry is null)
            {
                Announce(
                    report,
                    log,
                    "ASKED FOR WITH --live AND NOT SHOWN - " + (NvmlTelemetry.Unavailable ?? "no reason recorded") +
                    ". No fallback to the stub was made, deliberately: pass --live-stub to see the panels " +
                    "anyway, and read them knowing that its numbers are SYNTHETIC - invented on the spot, not " +
                    "measurements of this card. Nothing else about this run changed: the view is an observer " +
                    "and its absence costs no measurement.");
                return null;
            }

            var view = TryCreate(telemetry, state);
            if (view is null)
            {
                telemetry.Dispose();
                Announce(
                    report,
                    log,
                    "ASKED FOR WITH --live AND NOT SHOWN - " + (Unavailable ?? "no reason recorded") +
                    ". Run the probe in a real terminal and do not redirect its standard error. --live-stub " +
                    "does NOT help here: the stub replaces the sensors, not the terminal, so it is refused on " +
                    "the same line. Nothing else about this run changed.");
                return null;
            }

            Announce(
                report,
                log,
                "SHOWN during this run, driven by " + telemetry.SourceName +
                " It repaints between phases only, and is suspended for every warm-up round and every timed " +
                "repetition. That cadence is not a precaution, it is measured: a repaint between rounds moved " +
                "the C3 host arm's median by 33 to 41 % across three sittings at the quick shape - one box, " +
                "one shape, so read it as an order of magnitude and not as a constant - and the host arms are " +
                "the baseline every device ratio is divided by. Run --live-perturbation to reproduce it on " +
                "your own machine.");
            return view;
        }

        /// <summary>
        /// Writes one outcome to the report AND to the console, from one place.
        /// <para>
        /// One place because the two had already drifted: the report carried a reason for two of the three
        /// refusals and the console carried whichever of them <c>Program</c> happened to echo, while the
        /// third refusal - <c>--live</c> not passed at all - reached neither. A caller that has to remember
        /// to do both will eventually do one.
        /// </para>
        /// </summary>
        private static void Announce(Report report, TextWriter log, string note)
        {
            report.LiveViewNote = note;
            log.WriteLine("live view: " + note);
        }

        /// <summary>Runs <paramref name="body"/> with the display attached.</summary>
        public void Run(Action body)
        {
            _console.Live(Build()).Start(ctx =>
            {
                _context = ctx;
                try
                {
                    body();
                }
                finally
                {
                    _context = null;
                }
            });
        }

        /// <summary>
        /// Reads the sensors and repaints. <b>Call only from a point the caller knows is outside every
        /// timed region.</b> The read is a driver round trip.
        /// </summary>
        public void Refresh()
        {
            // FIRST, and before the suspension check - the order is the whole value of this guard, and it
            // was measured on 2026-08-22 rather than reasoned about. With the suspension tested first, a
            // repaint injected into the middle of Arm.TimeOnce was swallowed by it and the run reported
            // ZERO drops: the timed phases are exactly the phases the view is suspended for, so the one
            // call site this counter exists to catch was the one call site it could never see. A guard
            // that is silent precisely where the defect lives is worse than no guard, because the report
            // then carries a reassuring zero.
            //
            // The rule itself outranks everything else here. A sensor read is a driver round trip and the
            // repaint below allocates a frame; either one between a timestamp and its synchronise moves
            // that number and leaves the report looking entirely normal. Dropped AND counted.
            if (TimedRegion.IsInside)
            {
                DroppedInsideTimedRegion++;
                return;
            }

            if (Suspended || _context is null)
            {
                return;
            }

            _latest = _telemetry.Read();
            _context.UpdateTarget(Build());
            _context.Refresh();
            FramesPainted++;
        }

        /// <summary>Names what the probe is doing and repaints.</summary>
        public void Show(string phase)
        {
            _state.Phase = phase;
            Refresh();
        }

        /// <summary>Names the shape and what the probe is doing with it, and repaints.</summary>
        public void ShowCell(string cell, int n, string phase)
        {
            _state.Cell = cell;
            _state.N = n;
            _state.Phase = phase;
            Refresh();
        }

        /// <summary>Takes the completed cell's own numbers - never a second timer - and repaints.</summary>
        public void Completed(CellResult cell)
        {
            _state.RecordCompleted(cell);
            Refresh();
        }

        /// <summary>
        /// Paints one last frame naming the phase, then blocks every repaint until <see cref="Resume"/>.
        /// <para>
        /// The frame is painted BEFORE the freeze so the display says why it is about to stop moving.
        /// A view that simply stops looks like a hung probe, and a stranger watching a twenty-second
        /// warm-up would reasonably kill it.
        /// </para>
        /// </summary>
        public void FreezeFor(string phase)
        {
            Suspended = false;
            Show(phase);
            Suspended = true;
        }

        /// <summary>Lets the view repaint again after a timed phase.</summary>
        public void Resume() => Suspended = false;

        private IRenderable Build()
        {
            var grid = new Grid();
            grid.AddColumn();
            grid.AddRow(DevicePanel());
            grid.AddRow(ProbePanel());
            grid.AddRow(ThroughputPanel());
            return grid;
        }

        private IRenderable DevicePanel()
        {
            var table = new Table().Border(TableBorder.None).HideHeaders();
            table.AddColumn(new TableColumn("k").Width(12));
            table.AddColumn(new TableColumn("v"));

            if (_latest is null)
            {
                table.AddRow("card", "waiting for the first sample");
                return new Panel(table).Header("device").Expand();
            }

            // Escape() rather than trust the string: a card name containing '[' is valid markup to
            // Spectre and would throw mid-run. Overflow ellipsis handles a name wider than the panel.
            table.AddRow(
                new SpectreText("card"),
                new Markup(Markup.Escape(_latest.CardName)).Overflow(Overflow.Ellipsis));

            table.AddRow("memory", string.Create(
                CultureInfo.InvariantCulture,
                $"{Gib(_latest.VramUsedBytes)} used of {Gib(_latest.VramTotalBytes)} " +
                $"({_latest.VramUsedFraction.Format("%", "F1")}), device-wide, not this process"));

            table.AddRow("temperature", _latest.TemperatureCelsius.Format("C"));
            table.AddRow("utilisation", _latest.UtilisationPercent.Format("%"));
            table.AddRow("fan", _latest.FanPercent.Format("%"));
            return new Panel(table).Header("device").Expand();
        }

        private IRenderable ProbePanel()
        {
            var table = new Table().Border(TableBorder.None).HideHeaders();
            table.AddColumn(new TableColumn("k").Width(12));
            table.AddColumn(new TableColumn("v"));

            table.AddRow("shape", Markup.Escape(_state.N > 0 ? $"{_state.Cell}  n={_state.N}" : _state.Cell));
            table.AddRow("phase", Markup.Escape(_state.Phase));
            table.AddRow("warm-up", _state.WarmupRound > 0 ? $"round {_state.WarmupRound}" : "-");
            table.AddRow("progress", $"{_state.CellsDone} of {_state.CellsTotal} shape/batch combinations done");
            return new Panel(table).Header("probe").Expand();
        }

        private IRenderable ThroughputPanel()
        {
            if (_state.LastGflops.Count == 0)
            {
                return new Panel(new SpectreText(_state.CompletedLabel)).Header("throughput").Expand();
            }

            var table = new Table().Border(TableBorder.None);
            table.AddColumn("arm");
            table.AddColumn(new TableColumn("median ms").RightAligned());
            table.AddColumn(new TableColumn("GFLOP/s").RightAligned());

            foreach (var (arm, gflops) in _state.LastGflops)
            {
                table.AddRow(
                    Markup.Escape(arm),
                    string.Create(CultureInfo.InvariantCulture, $"{_state.LastMedianMs[arm]:F3}"),
                    string.Create(CultureInfo.InvariantCulture, $"{gflops:F1}"));
            }

            return new Panel(table)
                .Header($"throughput - last completed: {Markup.Escape(_state.CompletedLabel)}")
                .Expand();
        }

        private static string Gib(TelemetryReading bytes) => bytes.HasValue
            ? string.Create(CultureInfo.InvariantCulture, $"{bytes.Value / (1024.0 * 1024.0 * 1024.0):F2} GiB")
            : bytes.Reason;

        public void Dispose() => _telemetry.Dispose();
    }
}
