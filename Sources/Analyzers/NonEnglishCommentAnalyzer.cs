// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT037 — comments must be in English.
    ///
    /// <para>This codebase is published, and its comments carry more than description: measurements,
    /// rejected designs and the reason a thing is the way it is. A comment nobody outside one country can
    /// read is evidence that has been written down and then hidden.</para>
    ///
    /// <para><b>The detector is deliberately narrow, and that is the whole design.</b> A first attempt at
    /// measuring this by hand used a wider word list containing <c>pod</c>, <c>to</c>, <c>test</c> and
    /// <c>bo</c>; in a Kubernetes anomaly-detection codebase "pod" appears in most English comments, and the
    /// scan reported <b>154</b> violations where there were <b>4</b>. A rule that fires on correct code is
    /// suppressed within a week, and then it protects nothing — so the word list here holds only Polish
    /// words with no English homograph, and a single word is not enough to report.</para>
    ///
    /// <para>Two independent signals: any Polish diacritic, or two distinct words from the list. Comments
    /// that are <i>about</i> non-ASCII handling are exempt — there is a legitimate English comment in the
    /// BM25 tokenizer that lists Polish letters as the data it must not destroy.</para>
    ///
    /// <para>Severity is <b>Warning</b>, not Error, on purpose: the detector is a heuristic over natural
    /// language, and promoting a heuristic to a build error is how a tree acquires suppressions. Promote it
    /// once it has been quiet for a while on real changes.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class NonEnglishCommentAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT037";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Comment is not in English",
            messageFormat: "This comment appears not to be in English ({0}) — comments in this repository ship with it and are read by people who do not share your first language",
            category: "Design",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Comments here record measurements and rejected designs, not just description. Writing one in another language hides the evidence rather than the prose.");

        /// <summary>Letters that exist in Polish and not in English. Any one of them is decisive.</summary>
        private const string PolishLetters = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ";

        /// <summary>
        /// Polish words with <b>no</b> English homograph. Anything ambiguous is left out on purpose — see the
        /// 154-versus-4 note on the type. Adding a word that is also English silently converts this rule into
        /// noise.
        /// </summary>
        private static readonly HashSet<string> PolishWords = new(StringComparer.OrdinalIgnoreCase)
        {
            "jest", "sie", "nie", "tego", "ktory", "ktora", "ktore", "zeby", "wiec", "juz",
            "byl", "byla", "bylo", "moze", "trzeba", "musi", "wszystkie", "kazdy", "przez",
            "aby", "oraz", "tylko", "zawsze", "nigdy", "zalezy", "poniewaz", "dlatego",
            "jesli", "wtedy", "osobny", "osobne", "zapisuje", "inicjowane", "koncu",
            "wynik", "liczba", "zmiana", "stabilne", "szansa", "najlepsza",
        };

        /// <summary>A comment discussing non-ASCII handling legitimately contains non-ASCII letters.</summary>
        private static readonly string[] NonAsciiTopics =
        {
            "non-ascii", "diacritic", "unicode", "utf-8", "utf8", "codepoint", "code point", "encoding",
        };

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxTreeAction(AnalyzeTree);
        }

        private static void AnalyzeTree(SyntaxTreeAnalysisContext context)
        {
            var root = context.Tree.GetRoot(context.CancellationToken);

            foreach (var trivia in root.DescendantTrivia())
            {
                if (!IsComment(trivia.Kind()))
                {
                    continue;
                }

                var reason = Classify(trivia.ToString());

                if (reason is null)
                {
                    continue;
                }

                context.ReportDiagnostic(Diagnostic.Create(Rule, trivia.GetLocation(), reason));
            }
        }

        private static bool IsComment(SyntaxKind kind)
        {
            return kind is SyntaxKind.SingleLineCommentTrivia
                or SyntaxKind.MultiLineCommentTrivia
                or SyntaxKind.SingleLineDocumentationCommentTrivia
                or SyntaxKind.MultiLineDocumentationCommentTrivia;
        }

        /// <summary>Why this comment looks non-English, or <see langword="null"/> if it does not.</summary>
        private static string? Classify(string text)
        {
            if (MentionsNonAsciiHandling(text))
            {
                return null;
            }

            foreach (var ch in text)
            {
                if (PolishLetters.IndexOf(ch) >= 0)
                {
                    return "Polish letters";
                }
            }

            var words = DistinctPolishWords(text);

            // One word is noise; two independent ones are not a coincidence.
            if (words >= 2)
            {
                return words + " Polish words";
            }

            return null;
        }

        private static bool MentionsNonAsciiHandling(string text)
        {
            foreach (var topic in NonAsciiTopics)
            {
                if (text.IndexOf(topic, StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return true;
                }
            }

            return false;
        }

        private static int DistinctPolishWords(string text)
        {
            var found = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var start = -1;

            // BOUND: one pass over the comment text; length is the file's, not attacker-controlled.
            for (var i = 0; i <= text.Length; i++)
            {
                var isLetter = i < text.Length && char.IsLetter(text[i]);

                if (isLetter && start < 0)
                {
                    start = i;
                    continue;
                }

                if (isLetter || start < 0)
                {
                    continue;
                }

                var word = text.Substring(start, i - start);
                start = -1;

                if (PolishWords.Contains(word))
                {
                    found.Add(word);
                }
            }

            return found.Count;
        }
    }
}
