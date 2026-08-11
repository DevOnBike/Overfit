// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.MSBuild;

namespace DevOnBike.Overfit.Navigator
{
    /// <summary>
    /// Opens the Overfit solution into a Roslyn <see cref="Workspace"/> and holds it for the process lifetime.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The whole design of this tool turns on one measured fact: <b>opening the solution is expensive and
    /// producing the semantic model is far more expensive still</b>. <see cref="MSBuildWorkspace"/> shells out
    /// to MSBuild to evaluate every project, then Roslyn parses every source file; asking for a
    /// <see cref="Compilation"/> additionally binds every symbol and reads every referenced assembly. Both
    /// numbers are printed by the <c>measure</c> verb rather than assumed — see <c>docs/semantic-navigator.md</c>
    /// for what this box actually reports.
    /// </para>
    /// <para>
    /// Consequence: the loader is built to be created <b>once</b> and queried many times. Do not add a code path
    /// that constructs one per query.
    /// </para>
    /// </remarks>
    internal sealed class WorkspaceLoader : IDisposable
    {
        private readonly MSBuildWorkspace _workspace;
        private readonly List<string> _failures = new();
        private readonly IDisposable _failureSubscription;

        private WorkspaceLoader(MSBuildWorkspace workspace)
        {
            _workspace = workspace;
            _failureSubscription = workspace.RegisterWorkspaceFailedHandler(e => _failures.Add(e.Diagnostic.Message));
        }

        /// <summary>Solution opened into this workspace.</summary>
        public Solution Solution => _workspace.CurrentSolution;

        /// <summary>
        /// Non-fatal problems MSBuild or Roslyn reported while opening projects. A project that fails to load
        /// contributes <b>no symbols</b>, so a reference search over it silently returns fewer results rather
        /// than an error — which is why every caller is expected to surface this list.
        /// </summary>
        public IReadOnlyList<string> Failures => _failures;

        /// <summary>Wall-clock time spent inside <see cref="MSBuildWorkspace.OpenSolutionAsync"/>.</summary>
        public TimeSpan OpenDuration
        {
            get; private set;
        }

        /// <summary>Opens <paramref name="solutionPath"/>, registering MSBuild first if nothing has yet.</summary>
        public static async Task<WorkspaceLoader> OpenAsync(string solutionPath, CancellationToken cancellationToken)
        {
            EnsureMsBuildRegistered();

            var workspace = MSBuildWorkspace.Create();
            var loader = new WorkspaceLoader(workspace);

            var started = Stopwatch.GetTimestamp();
            await workspace.OpenSolutionAsync(solutionPath, cancellationToken: cancellationToken).ConfigureAwait(false);
            loader.OpenDuration = Stopwatch.GetElapsedTime(started);

            return loader;
        }

        /// <summary>
        /// Forces the semantic model for every C# project, so later queries do not pay for it one at a time.
        /// </summary>
        /// <remarks>
        /// Returns per-project durations because they are wildly uneven here — Main dominates, and knowing that
        /// is what makes a single-project query worth offering separately from a solution-wide one.
        /// </remarks>
        public async Task<IReadOnlyList<(string Project, TimeSpan Duration, int Documents)>> WarmCompilationsAsync(
            CancellationToken cancellationToken)
        {
            var results = new List<(string, TimeSpan, int)>();

            foreach (var project in Solution.Projects)
            {
                if (project.Language != LanguageNames.CSharp)
                {
                    continue;
                }

                var started = Stopwatch.GetTimestamp();
                await project.GetCompilationAsync(cancellationToken).ConfigureAwait(false);
                results.Add((project.Name, Stopwatch.GetElapsedTime(started), project.Documents.Count()));
            }

            return results;
        }

        /// <summary>
        /// Registers an MSBuild instance. Must run before any MSBuild type is loaded, which is why the workspace
        /// is only ever touched from a separate non-inlined method.
        /// </summary>
        private static void EnsureMsBuildRegistered()
        {
            if (Microsoft.Build.Locator.MSBuildLocator.IsRegistered)
            {
                return;
            }

            Microsoft.Build.Locator.MSBuildLocator.RegisterDefaults();
        }

        public void Dispose()
        {
            _failureSubscription.Dispose();
            _workspace.Dispose();
        }
    }
}
