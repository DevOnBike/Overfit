// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// Turns every <c>**/Schemas/*.json</c> into a const on <c>OverfitSchemas</c>.
    ///
    /// <para><b>Replaces the <c>EmbedJsonSchemas</c> MSBuild task</b>, which was forty lines of C# embedded in
    /// <c>Main.csproj</c> as a <c>RoslynCodeTaskFactory</c> fragment — code with no IDE, no debugger, no
    /// tests and no compiler checking it until a build failed. Codegen is what a source generator is for;
    /// the MSBuild version existed because it was written before anybody reached for one.</para>
    ///
    /// <para><b>Three things improve by moving, not just the location.</b> The escaping is done by Roslyn's
    /// own literal formatter instead of by doubling quote characters in a verbatim string, so a schema
    /// containing a backslash or a newline can no longer produce source that does not compile. The generated
    /// file is incremental — it re-runs when a schema changes rather than on every build. And the const name
    /// is now checked for being a valid C# identifier, which the task never did: a file called
    /// <c>tool-call.json</c> used to emit <c>public const string tool-call</c> and fail with a syntax error
    /// in generated code nobody wrote.</para>
    ///
    /// <para>Nothing is emitted when a project has no schema files, so this costs nothing in the projects
    /// that do not use it — which, after the anomaly guard was split out, is most of them.</para>
    /// </summary>
    [Generator]
    public sealed class OverfitSchemasGenerator : IIncrementalGenerator
    {
        /// <summary>Two schema files whose names collide on the single generated class.</summary>
        internal static readonly DiagnosticDescriptor DuplicateName = new(
            "OVERFIT035",
            title: "Two schema files map to the same generated constant",
            messageFormat: "Schema file names must be unique across all Schemas/ folders: '{0}' collides with an earlier file on the constant 'OverfitSchemas.{0}'",
            category: "Design",
            defaultSeverity: DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Every Schemas/*.json collapses into one class, so the file name IS the constant name. A collision would otherwise surface as a duplicate-member error in generated code the author never wrote.");

        /// <summary>A schema file whose name cannot be a C# identifier.</summary>
        internal static readonly DiagnosticDescriptor UnusableName = new(
            "OVERFIT036",
            title: "Schema file name is not a usable constant name",
            messageFormat: "'{0}' cannot become a C# identifier, so it cannot be emitted as a constant — rename the file",
            category: "Design",
            defaultSeverity: DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "The generated constant takes the file name verbatim. A name containing a hyphen or starting with a digit produces source that does not compile, reported at a line the author did not write.");

        public void Initialize(IncrementalGeneratorInitializationContext context)
        {
            var schemas = context.AdditionalTextsProvider
                .Where(static text => IsSchema(text.Path))
                .Select(static (text, token) => (Name: NameOf(text.Path), Body: text.GetText(token)?.ToString()))
                .Collect();

            context.RegisterSourceOutput(schemas, Emit);
        }

        private static bool IsSchema(string path)
        {
            if (!path.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            // Directory-scoped on purpose: AdditionalFiles is a shared channel and other tooling puts json in
            // it. Only files under a folder called Schemas are schemas.
            var separator = path.LastIndexOfAny(['/', '\\']);

            if (separator <= 0)
            {
                return false;
            }

            var directory = path.Substring(0, separator);
            var leaf = directory.Substring(directory.LastIndexOfAny(['/', '\\']) + 1);

            return string.Equals(leaf, "Schemas", StringComparison.OrdinalIgnoreCase);
        }

        private static string NameOf(string path)
        {
            var separator = path.LastIndexOfAny(['/', '\\']);
            var file = separator >= 0 ? path.Substring(separator + 1) : path;
            var dot = file.LastIndexOf('.');

            return dot > 0 ? file.Substring(0, dot) : file;
        }

        private static bool IsUsableIdentifier(string name)
        {
            if (name.Length == 0 || char.IsDigit(name[0]))
            {
                return false;
            }

            foreach (var c in name)
            {
                if (!char.IsLetterOrDigit(c) && c != '_')
                {
                    return false;
                }
            }

            return true;
        }

        private static void Emit(
            SourceProductionContext context,
            ImmutableArray<(string Name, string? Body)> schemas)
        {
            if (schemas.IsDefaultOrEmpty)
            {
                return;
            }

            var text = new StringBuilder();

            text.AppendLine("// <auto-generated/>");
            text.AppendLine("// Generated from Schemas/*.json by OverfitSchemasGenerator. Do not edit.");
            text.AppendLine("namespace DevOnBike.Overfit.Schemas");
            text.AppendLine("{");
            text.AppendLine("    internal static class OverfitSchemas");
            text.AppendLine("    {");

            var seen = new HashSet<string>(StringComparer.Ordinal);
            var emitted = 0;

            // Sorted, so the generated file does not change when the file system enumerates differently —
            // a build output that reorders itself defeats incremental compilation and pollutes diffs.
            foreach (var (name, body) in schemas.Sort(static (a, b) => string.CompareOrdinal(a.Name, b.Name)))
            {
                if (body is null)
                {
                    continue;
                }

                if (!IsUsableIdentifier(name))
                {
                    context.ReportDiagnostic(Diagnostic.Create(UnusableName, Location.None, name));

                    continue;
                }

                if (!seen.Add(name))
                {
                    context.ReportDiagnostic(Diagnostic.Create(DuplicateName, Location.None, name));

                    continue;
                }

                // Roslyn's own literal formatter rather than doubling quotes into a verbatim string: it
                // handles backslashes, newlines and anything else a JSON file can legally contain.
                text.Append("        public const string ").Append(name).Append(" = ")
                    .Append(SymbolDisplay.FormatLiteral(body, quote: true)).AppendLine(";");

                emitted++;
            }

            text.AppendLine("    }");
            text.AppendLine("}");

            if (emitted > 0)
            {
                context.AddSource("OverfitSchemas.g.cs", SourceText.From(text.ToString(), Encoding.UTF8));
            }
        }
    }
}
