// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// Compiles a C# snippet to an assembly image in memory, so the comparator's tests need no fixture.
    ///
    /// <para><b>Why not compare two real packages.</b> CI is Linux with no models and no populated NuGet cache;
    /// this repository already has a recorded failure mode where a fixture-dependent test silently skips
    /// forever and reads as coverage. Generating both inputs at test time means the expected answer is known by
    /// construction — "these two differ in exactly one statement" is not an observation about a downloaded
    /// file, it is the definition of the input.</para>
    ///
    /// <para><b>Non-deterministic emit on purpose.</b> <c>WithDeterministic(false)</c> makes Roslyn stamp a
    /// fresh MVID and timestamp into every build, which is exactly the real-world condition the comparator has
    /// to see through: two builds of identical source that differ as files. Turning determinism on would make
    /// the most important test in the suite pass for the wrong reason.</para>
    /// </summary>
    internal static class TinyAssemblyCompiler
    {
        private static readonly Lazy<IReadOnlyList<MetadataReference>> References =
            new Lazy<IReadOnlyList<MetadataReference>>(BuildReferences);

        /// <summary>Compiles <paramref name="source"/> and returns the assembly image.</summary>
        /// <param name="assemblyName">
        /// Kept identical across the two sides of a comparison unless a test is specifically about the assembly
        /// identity — a differing name would move the AssemblyDef row and muddy every other assertion.
        /// </param>
        internal static byte[] Compile(string source, string assemblyName = "TinyAssembly")
        {
            ArgumentNullException.ThrowIfNull(source);

            var tree = CSharpSyntaxTree.ParseText(source);
            var options = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
                .WithOptimizationLevel(OptimizationLevel.Release)
                .WithDeterministic(false);

            var compilation = CSharpCompilation.Create(assemblyName, [tree], References.Value, options);

            using var stream = new MemoryStream();

            var result = compilation.Emit(stream);

            // A snippet that does not compile would produce an empty image, and an empty image compares equal
            // to another empty image — a green test that proves nothing. It fails loudly here instead.
            if (!result.Success)
            {
                var errors = new List<string>();

                // BOUND: one iteration per diagnostic.
                foreach (var diagnostic in result.Diagnostics)
                {
                    if (diagnostic.Severity == DiagnosticSeverity.Error)
                    {
                        errors.Add(diagnostic.ToString());
                    }
                }

                throw new InvalidOperationException(
                    "the test snippet does not compile: " + string.Join("; ", errors));
            }

            return stream.ToArray();
        }

        private static IReadOnlyList<MetadataReference> BuildReferences()
        {
            var references = new List<MetadataReference>
            {
                MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
            };

            var trusted = AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") as string ?? string.Empty;

            // Deliberately a short list rather than the whole platform: every entry is a file read, and the
            // snippets here use nothing beyond primitives and the assembly-level attributes.
            var wanted = new[] { "System.Runtime.dll", "System.Runtime.Extensions.dll" };

            // BOUND: one iteration per entry of the trusted platform assemblies list.
            foreach (var path in trusted.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
            {
                var fileName = Path.GetFileName(path);

                // BOUND: one iteration per wanted name.
                foreach (var name in wanted)
                {
                    if (string.Equals(fileName, name, StringComparison.OrdinalIgnoreCase))
                    {
                        references.Add(MetadataReference.CreateFromFile(path));
                    }
                }
            }

            return references;
        }
    }
}
