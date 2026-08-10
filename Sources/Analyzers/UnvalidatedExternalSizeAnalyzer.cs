// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT038 — a count read from a file sizes an allocation or bounds a loop without being validated first.
    /// This is the semantic half of NASA's Power of 10 rule 2; <see cref="UnboundedLoopAnalyzer"/> (OVERFIT023)
    /// is the syntactic half, and banning <c>while (true)</c> alone leaves the larger hole open.
    ///
    /// <para><b>The bound is the attacker's, not the author's.</b> Every loader in this library reads model files
    /// somebody else produced — GGUF, ONNX, safetensors, ggml, a LoRA adapter. A four-byte length field in such a
    /// file is a number chosen by whoever wrote the file, and <c>new string[nTokens]</c> on it is that number
    /// turned into an allocation request. A header declaring two billion entries costs no space in the file and
    /// terminates the host process, which is not a failure the caller can catch or report.</para>
    ///
    /// <para><b>Measured before it was written, which is the only reason it exists.</b> A crude text
    /// approximation over <c>Sources/</c> on 2026-08-02 found 19 read-then-use sites in the whole tree: 8 already
    /// guarded, 11 flagged, and after reading each one, <b>7 real defects</b> — four of them fifteen consecutive
    /// lines of a ggml reader with no bound on anything. Nineteen sites is not a rule that buries anyone.</para>
    ///
    /// <para><b>Ordering is the part a human eye skips, and it is why this is a Roslyn rule rather than a review
    /// question.</b> All four false positives in that sweep were one shape — <c>if (length != expected) throw</c>
    /// — which ordinary local data flow sees. But the defect no review had found was the opposite: in
    /// <c>ModelSerializer</c> the guard existed and ran <b>after</b> the array had already been allocated and
    /// filled. A reader sees the check a few lines below and marks it done. This rule only accepts a validator
    /// that appears between the read and the first use.</para>
    ///
    /// <para><b>What counts as validating it.</b> An <c>if</c> whose condition mentions the value and whose body
    /// throws or returns — any comparison, equality included; a call whose name begins with a guard verb
    /// (<c>Require</c>, <c>Validate</c>, <c>Ensure</c>, <c>Check</c>, <c>Throw</c>, <c>Assert</c>, <c>Guard</c>),
    /// which covers <c>ArgumentOutOfRangeException.ThrowIfNegative</c> and the house <c>Require…</c> helpers; or
    /// capping the value outright with <c>Math.Min</c> / <c>Math.Clamp</c>.</para>
    ///
    /// <para><b>What it cannot see, inherited from the sweep.</b> Local flow finds "read → use" inside one
    /// method; a bound established two calls away reads as a violation. All nineteen sites happened to be
    /// single-method, so the question did not arise — but it will, and that is the residual false-positive risk.
    /// The escape hatch is the same per-site contract <c>OVERFIT022</c>/<c>OVERFIT023</c> use: a
    /// <c>#pragma warning disable OVERFIT038</c> whose comment names where the bound is actually established.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class UnvalidatedExternalSizeAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT038";

        /// <summary>
        /// Types whose read methods return a number chosen by whoever wrote the file. Matched by full name so a
        /// wrapper of our own — which has already had the chance to validate — is not swept in by accident.
        /// </summary>
        private static readonly HashSet<string> UntrustedReaderTypes = new(StringComparer.Ordinal)
        {
            "System.IO.BinaryReader",
            "System.Buffers.Binary.BinaryPrimitives",
            "System.Text.Json.JsonElement",
            "System.Text.Json.Utf8JsonReader",
        };

        /// <summary>
        /// Fallback for a snippet whose reader type does not resolve. Names only, so it is deliberately narrow:
        /// every entry is a fixed-width integer read, never a string or a payload read.
        /// </summary>
        private static readonly HashSet<string> UntrustedReadMethodNames = new(StringComparer.Ordinal)
        {
            "ReadInt16", "ReadUInt16", "ReadInt32", "ReadUInt32", "ReadInt64", "ReadUInt64",
            "ReadByte", "ReadSByte", "Read7BitEncodedInt", "Read7BitEncodedInt64",
            "ReadInt16LittleEndian", "ReadUInt16LittleEndian", "ReadInt32LittleEndian",
            "ReadUInt32LittleEndian", "ReadInt64LittleEndian", "ReadUInt64LittleEndian",
            "ReadInt16BigEndian", "ReadUInt16BigEndian", "ReadInt32BigEndian",
            "ReadUInt32BigEndian", "ReadInt64BigEndian", "ReadUInt64BigEndian",
            "GetInt16", "GetUInt16", "GetInt32", "GetUInt32", "GetInt64", "GetUInt64",
        };

        /// <summary>
        /// Reads that report a size the parser has already materialised rather than one the file declares.
        ///
        /// <para><b>Declared and measured are not the same number, and the difference is the whole rule.</b>
        /// <c>JsonElement.GetArrayLength()</c> counts elements that are already in the document, so it is
        /// bounded by the bytes that were accepted; a declared count is bounded by nothing, which is why four
        /// bytes in a header can ask for sixteen gigabytes. Excluded after this rule's first inventory called
        /// <c>XgboostModelLoader</c>'s tree count a defect.</para>
        /// </summary>
        private static readonly HashSet<string> MeasuredLengths = new(StringComparer.Ordinal)
        {
            "GetArrayLength", "GetPropertyCount",
        };

        /// <summary>A call beginning with one of these is the author asserting a bound, wherever it lives.</summary>
        private static readonly string[] GuardVerbs =
        [
            "Require", "Validate", "Ensure", "Check", "Throw", "Assert", "Guard", "Verify",
        ];

        /// <summary>Capping the value is as good as rejecting it — the allocation can no longer be chosen.</summary>
        private static readonly HashSet<string> CappingMethods = new(StringComparer.Ordinal)
        {
            "Min", "Clamp",
        };

        /// <summary>Reads that consume a caller-supplied count and are therefore sinks, not sources.</summary>
        private static readonly HashSet<string> CountedReadMethods = new(StringComparer.Ordinal)
        {
            "ReadBytes", "ReadChars",
        };

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Externally-read count sizes an allocation or bounds a loop without validation",
            messageFormat: "'{0}' comes from a file and {1} with no bound checked in between — a header declaring a huge count becomes an allocation request the caller cannot catch; validate it before this line",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A length field in a model file is a number chosen by whoever wrote the file. Sizing an array, a stackalloc, a counted read or a loop on it hands the host process to the file. The validator must run BEFORE the first use: a check that appears a few lines below reads as done to a human and is not.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();

            context.RegisterSyntaxNodeAction(
                AnalyzeAllocation,
                SyntaxKind.ArrayCreationExpression,
                SyntaxKind.StackAllocArrayCreationExpression);

            context.RegisterSyntaxNodeAction(AnalyzeCountedRead, SyntaxKind.InvocationExpression);
            context.RegisterSyntaxNodeAction(AnalyzeForLoop, SyntaxKind.ForStatement);
            context.RegisterSyntaxNodeAction(AnalyzeWhileLoop, SyntaxKind.WhileStatement);
        }

        private static void AnalyzeAllocation(SyntaxNodeAnalysisContext context)
        {
            var sizes = context.Node switch
            {
                ArrayCreationExpressionSyntax { Type.RankSpecifiers.Count: > 0 } array
                    => array.Type.RankSpecifiers[0].Sizes,
                StackAllocArrayCreationExpressionSyntax { Type: ArrayTypeSyntax { RankSpecifiers.Count: > 0 } type }
                    => type.RankSpecifiers[0].Sizes,
                _ => default,
            };

            if (sizes.Count == 0)
            {
                return;
            }

            var verb = context.Node.IsKind(SyntaxKind.StackAllocArrayCreationExpression)
                ? "sizes a stackalloc"
                : "sizes an array";

            foreach (var size in sizes)
            {
                if (size is OmittedArraySizeExpressionSyntax)
                {
                    continue;
                }

                Inspect(context, size, context.Node, verb);
            }
        }

        private static void AnalyzeCountedRead(SyntaxNodeAnalysisContext context)
        {
            var invocation = (InvocationExpressionSyntax)context.Node;

            if (!CountedReadMethods.Contains(SimpleNameOf(invocation)))
            {
                return;
            }

            if (invocation.ArgumentList.Arguments.Count != 1)
            {
                return;
            }

            Inspect(context, invocation.ArgumentList.Arguments[0].Expression, invocation, "is read as a byte count");
        }

        private static void AnalyzeForLoop(SyntaxNodeAnalysisContext context)
        {
            var loop = (ForStatementSyntax)context.Node;

            if (loop.Condition is null)
            {
                return;
            }

            Inspect(context, loop.Condition, loop, "bounds a loop");
        }

        private static void AnalyzeWhileLoop(SyntaxNodeAnalysisContext context)
        {
            var loop = (WhileStatementSyntax)context.Node;

            Inspect(context, loop.Condition, loop, "bounds a loop");
        }

        /// <summary>
        /// Reports every externally-read value reaching <paramref name="expression"/> that nothing bounded
        /// between the read and <paramref name="sink"/>.
        /// </summary>
        private static void Inspect(
            SyntaxNodeAnalysisContext context,
            ExpressionSyntax expression,
            SyntaxNode sink,
            string verb)
        {
            // A cap inside the size expression itself settles the question, whatever fed it.
            if (ContainsCappingCall(expression))
            {
                return;
            }

            // Read straight into the sink — `new byte[br.ReadInt32()]`. Nothing could have validated it, so no
            // window needs searching.
            foreach (var node in expression.DescendantNodesAndSelf())
            {
                if (node is InvocationExpressionSyntax direct && IsUntrustedRead(context, direct))
                {
                    Report(context, sink, direct.ToString(), verb);
                    return;
                }
            }

            var body = EnclosingBody(sink);

            if (body is null)
            {
                return;
            }

            foreach (var local in TaintedLocals(context, expression, body))
            {
                if (HasBoundBetween(body, local.Name, local.ReadEnd, sink.SpanStart))
                {
                    continue;
                }

                Report(context, sink, local.Name, verb);
            }
        }

        private static void Report(SyntaxNodeAnalysisContext context, SyntaxNode sink, string name, string verb)
        {
            context.ReportDiagnostic(Diagnostic.Create(Rule, sink.GetLocation(), name, verb));
        }

        /// <summary>
        /// Locals used by <paramref name="expression"/> whose value originates in an untrusted read, together
        /// with the position that read finishes at — the start of the window a validator has to appear in.
        ///
        /// <para>Worklist rather than recursion: one assignment hop (<c>var n = header.Count;</c>) is common
        /// enough to be worth following, and OVERFIT022 forbids doing it the obvious way.</para>
        /// </summary>
        private static List<(string Name, int ReadEnd)> TaintedLocals(
            SyntaxNodeAnalysisContext context,
            ExpressionSyntax expression,
            SyntaxNode body)
        {
            var found = new List<(string, int)>();
            var seen = new HashSet<string>(StringComparer.Ordinal);
            var pending = new Queue<IdentifierNameSyntax>();

            foreach (var node in expression.DescendantNodesAndSelf())
            {
                if (node is IdentifierNameSyntax identifier)
                {
                    pending.Enqueue(identifier);
                }
            }

            var hops = 0;

            while (pending.Count > 0 && hops < 64)
            {
                hops++;
                var identifier = pending.Dequeue();
                var name = identifier.Identifier.ValueText;

                if (!seen.Add(name))
                {
                    continue;
                }

                if (context.SemanticModel.GetSymbolInfo(identifier, context.CancellationToken).Symbol
                    is not ILocalSymbol)
                {
                    continue;
                }

                var initializer = InitializerOf(body, name);

                if (initializer is null)
                {
                    continue;
                }

                var tainted = false;

                foreach (var node in initializer.DescendantNodesAndSelf())
                {
                    if (node is InvocationExpressionSyntax invocation && IsUntrustedRead(context, invocation))
                    {
                        tainted = true;
                        break;
                    }
                }

                if (tainted)
                {
                    // A cap applied at the point of the read is the bound, and it is the cheapest form of it.
                    if (!ContainsCappingCall(initializer))
                    {
                        found.Add((name, initializer.Span.End));
                    }

                    continue;
                }

                foreach (var node in initializer.DescendantNodesAndSelf())
                {
                    if (node is IdentifierNameSyntax hop)
                    {
                        pending.Enqueue(hop);
                    }
                }
            }

            return found;
        }

        /// <summary>The initialiser of the first declaration of <paramref name="name"/> inside the body.</summary>
        private static ExpressionSyntax? InitializerOf(SyntaxNode body, string name)
        {
            foreach (var declarator in body.DescendantNodes().OfType<VariableDeclaratorSyntax>())
            {
                if (declarator.Identifier.ValueText != name)
                {
                    continue;
                }

                return declarator.Initializer?.Value;
            }

            return null;
        }

        /// <summary>
        /// Whether something between <paramref name="from"/> and <paramref name="to"/> establishes a bound on
        /// <paramref name="name"/>. The window is half the rule: a guard outside it is the ModelSerializer
        /// defect, where the array was allocated and filled before anything asked whether the count was sane.
        /// </summary>
        private static bool HasBoundBetween(SyntaxNode body, string name, int from, int to)
        {
            foreach (var node in body.DescendantNodes())
            {
                if (node.SpanStart < from || node.SpanStart >= to)
                {
                    continue;
                }

                if (node is IfStatementSyntax branch
                    && Mentions(branch.Condition, name)
                    && Escapes(branch.Statement))
                {
                    return true;
                }

                if (node is not InvocationExpressionSyntax invocation)
                {
                    continue;
                }

                if (!Mentions(invocation.ArgumentList, name))
                {
                    continue;
                }

                var called = SimpleNameOf(invocation);

                if (CappingMethods.Contains(called))
                {
                    return true;
                }

                foreach (var verb in GuardVerbs)
                {
                    if (called.StartsWith(verb, StringComparison.Ordinal))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        /// <summary>Whether the branch leaves — a guard that only logs is not a bound.</summary>
        private static bool Escapes(SyntaxNode? statement)
        {
            if (statement is null)
            {
                return false;
            }

            foreach (var node in statement.DescendantNodesAndSelf())
            {
                if (node is ThrowStatementSyntax or ThrowExpressionSyntax or ReturnStatementSyntax)
                {
                    return true;
                }
            }

            return false;
        }

        private static bool Mentions(SyntaxNode? node, string name)
        {
            if (node is null)
            {
                return false;
            }

            foreach (var descendant in node.DescendantNodesAndSelf())
            {
                if (descendant is IdentifierNameSyntax identifier
                    && identifier.Identifier.ValueText == name)
                {
                    return true;
                }
            }

            return false;
        }

        private static bool ContainsCappingCall(SyntaxNode node)
        {
            foreach (var descendant in node.DescendantNodesAndSelf())
            {
                if (descendant is InvocationExpressionSyntax invocation
                    && CappingMethods.Contains(SimpleNameOf(invocation)))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Whether the invocation returns a number taken verbatim from a file. Semantic first, so a
        /// <c>Read…</c> method of our own that already validates is not swept in; the name-only fallback exists
        /// for snippets and for code the model cannot resolve.
        /// </summary>
        private static bool IsUntrustedRead(SyntaxNodeAnalysisContext context, InvocationExpressionSyntax invocation)
        {
            var name = SimpleNameOf(invocation);

            if (MeasuredLengths.Contains(name))
            {
                return false;
            }

            // A read that does not return a number cannot be a count, and the exclusion is not academic: the
            // standard JSON pull loop is `while (reader.Read())`, whose Read returns bool and means "is there
            // more", not "how many". Thirteen of the first thirty sites this rule reported were that line, in
            // six readers — a third of the output, all of it noise, and the kind of ratio that gets a rule
            // suppressed wholesale rather than fixed.
            if (!IsCountLike(context, invocation))
            {
                return false;
            }

            if (context.SemanticModel.GetSymbolInfo(invocation, context.CancellationToken).Symbol
                is IMethodSymbol method)
            {
                var owner = method.ContainingType?.ToDisplayString();

                if (owner is not null && UntrustedReaderTypes.Contains(owner))
                {
                    return name.StartsWith("Read", StringComparison.Ordinal)
                        || name.StartsWith("Get", StringComparison.Ordinal);
                }

                // Resolved, and not one of the untrusted readers — an owned wrapper has had its chance to
                // validate, and treating it as a source would flag the very helpers this rule asks for.
                return false;
            }

            return UntrustedReadMethodNames.Contains(name);
        }

        /// <summary>
        /// Whether the call returns an integer. Unresolved types are treated as counts, so the name-only
        /// fallback keeps working on a snippet — every name in that list is a fixed-width integer read.
        /// </summary>
        private static bool IsCountLike(SyntaxNodeAnalysisContext context, InvocationExpressionSyntax invocation)
        {
            var type = context.SemanticModel.GetTypeInfo(invocation, context.CancellationToken).Type;

            if (type is null || type.TypeKind == TypeKind.Error)
            {
                return true;
            }

            return type.SpecialType is SpecialType.System_Byte
                or SpecialType.System_SByte
                or SpecialType.System_Int16
                or SpecialType.System_UInt16
                or SpecialType.System_Int32
                or SpecialType.System_UInt32
                or SpecialType.System_Int64
                or SpecialType.System_UInt64
                or SpecialType.System_IntPtr
                or SpecialType.System_UIntPtr;
        }

        private static string SimpleNameOf(InvocationExpressionSyntax invocation)
        {
            return invocation.Expression switch
            {
                MemberAccessExpressionSyntax member => member.Name.Identifier.ValueText,
                IdentifierNameSyntax identifier => identifier.Identifier.ValueText,
                MemberBindingExpressionSyntax binding => binding.Name.Identifier.ValueText,
                _ => string.Empty,
            };
        }

        /// <summary>The body the sink lives in — the scope a validator has to be found inside.</summary>
        private static SyntaxNode? EnclosingBody(SyntaxNode node)
        {
            SyntaxNode? outermost = null;

            for (var current = node.Parent; current is not null; current = current.Parent)
            {
                if (current is BlockSyntax or ArrowExpressionClauseSyntax)
                {
                    outermost = current;
                }

                if (current is MemberDeclarationSyntax or LocalFunctionStatementSyntax or AnonymousFunctionExpressionSyntax)
                {
                    return outermost;
                }
            }

            return outermost;
        }
    }
}
