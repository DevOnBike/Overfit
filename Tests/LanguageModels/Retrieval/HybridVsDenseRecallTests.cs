// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Embeddings;
using DevOnBike.Overfit.LanguageModels.Retrieval;
using DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Retrieval
{
    /// <summary>
    /// Measures what hybrid retrieval actually bought, on <b>real MiniLM embeddings</b>, using the same
    /// <see cref="RagEvaluator"/> harness a customer would point at their own corpus — rather than asserting
    /// that hybrid must be better because the mechanism sounds convincing.
    ///
    /// <para><b>The result depends on the query mix, and that is the finding, not a flaw.</b> Cases are split
    /// into three deliberately balanced groups — five purely semantic (the target shares no distinctive
    /// vocabulary with the question, so only the dense arm can find it), five identifier lookups (error codes,
    /// policy numbers, form numbers), and five natural mixed questions. Reporting one blended number would
    /// hide the trade-off; reporting the three separately shows exactly what each arm contributes and what
    /// hybrid costs on the semantic side.</para>
    ///
    /// <para>The corpus is synthetic but written to resemble the target domain: an insurance/support knowledge
    /// base where prose and hard identifiers sit side by side. That shape is the claim being tested — on a
    /// corpus of pure prose with no identifiers, expect hybrid to gain nothing.</para>
    ///
    /// <para><b>MEASURED RESULT (2026-07-21, MiniLM, 18 chunks, recall@5) — hybrid was a NET REGRESSION here,
    /// which is why <c>McpRagIndex</c> was NOT switched over to it:</b></para>
    /// <code>
    /// group         dense R@K  hybrid R@K  dense MRR  hybrid MRR
    /// semantic           1.00        0.80      0.900       0.700   &lt;- regression
    /// identifier         1.00        1.00      0.800       0.900   &lt;- the predicted gain, confirmed
    /// mixed              1.00        1.00      1.000       1.000
    /// OVERALL            1.00        0.93      0.900       0.867
    /// </code>
    ///
    /// <para><b>Diagnosed cause.</b> "how long until my case is resolved?" fell out of the top-5 entirely
    /// because the lexical arm ranked <c>data-retention</c> ("<i>How long</i> we keep records") first, matching
    /// on the function words <i>how</i> and <i>long</i> alone. <see cref="Bm25Index.Tokenize"/> deliberately
    /// applies no stop-word list, and on an 18-document corpus IDF cannot compensate: "how" occurs in 2 of 18
    /// documents, so it earns a HIGH idf. On a realistic corpus of hundreds-to-thousands of chunks those words
    /// appear nearly everywhere and their idf collapses toward zero, so this failure mode should largely
    /// disappear.</para>
    ///
    /// <para><b>That prediction was then confirmed and this result superseded.</b>
    /// <see cref="HybridVsDenseOnDocsCorpusTests"/> re-ran the same comparison on a real 481-chunk corpus and
    /// hybrid won everywhere — recall@5 0.61 → 0.89, MRR 0.500 → 0.736, including the semantic group. So the
    /// regression measured here was an artefact of an 18-chunk corpus, not a property of hybrid retrieval.
    /// <b>Keep this test as the small-corpus caveat</b> (hybrid genuinely can hurt when the corpus is tiny),
    /// but do not treat it as evidence about the technique.</para>
    /// </summary>
    public sealed class HybridVsDenseRecallTests
    {
        private readonly ITestOutputHelper _out;

        public HybridVsDenseRecallTests(ITestOutputHelper output) => _out = output;

        private static readonly (string Id, string Text)[] Corpus =
        [
            ("policy-cancel", "Ending your cover. You may terminate the agreement at any time by giving fourteen days written notice to the address on your schedule. Any premium paid for the remaining period is refunded pro rata."),
            ("policy-renew", "Automatic continuation. Unless you tell us otherwise, the agreement rolls over for a further twelve months on its anniversary and the premium is recalculated."),
            ("claims-howto", "Reporting an incident. Contact us within seven days of becoming aware of the event. You will be asked for photographs and, where relevant, the police reference."),
            ("claims-time", "Settlement timing. Straightforward matters are concluded within thirty days of the last document we request. Complex matters may take longer."),
            ("billing-instalments", "Paying monthly. The premium may be spread across twelve instalments collected by direct debit on the fifth working day of each month."),
            ("billing-arrears", "Missed payments. If a collection fails we retry once. After two failures the cover is suspended and a notice is issued."),
            ("err-4021", "Error E-4021 is returned when the uploaded document exceeds the size limit. Compress the file below 10 MB and retry the upload."),
            ("err-4022", "Error E-4022 is returned when the uploaded document has an unsupported type. Convert the file to PDF and retry the upload."),
            ("err-5310", "Error E-5310 indicates the signing service was unreachable. The request is retried automatically for one hour before it is abandoned."),
            ("policy-40021", "Policy PL-88-40021 covers the commercial fleet of Northwind Logistics. The schedule lists eighteen vehicles and a named-driver restriction."),
            ("policy-40022", "Policy PL-88-40022 covers the commercial fleet of Contoso Freight. The schedule lists four vehicles with unrestricted driving."),
            ("form-cl17", "Form CL-17 is the claim notification form for goods in transit. Submit it with the consignment note and the carrier's damage report."),
            ("form-cl18", "Form CL-18 is the claim notification form for warehouse stock. Submit it with the stock ledger extract."),
            ("contact-hours", "When we are open. The service desk answers calls between eight in the morning and six in the evening, Monday to Friday, excluding public holidays."),
            ("data-retention", "How long we keep records. Documents relating to an agreement are retained for six years after it ends, after which they are destroyed securely."),
            ("excess", "The amount you pay. Each accepted claim carries a fixed contribution deducted from the settlement, shown on your schedule."),
            ("no-claims", "Discount for a clean record. Each consecutive year without an accepted claim increases the reduction applied at renewal, up to a maximum after five years."),
            ("drivers", "Who may drive. Only individuals listed on the schedule may operate the vehicles, unless the agreement states unrestricted driving."),
        ];

        // Purely semantic: the question deliberately shares no distinctive term with its target, so BM25
        // cannot possibly find it and only the dense arm can. This is the group hybrid could damage.
        private static RetrievalCase[] SemanticCases() =>
        [
            new("how do I stop my insurance?", "policy-cancel"),
            new("how long until my case is resolved?", "claims-time"),
            new("what happens if my card is declined?", "billing-arrears"),
            new("do I get a reward for being safe?", "no-claims"),
            new("when can I reach somebody by phone?", "contact-hours"),
        ];

        // Identifier lookups: the exact token IS the query. This is the group dense retrieval is bad at,
        // because neighbouring codes are near-identical directions in embedding space.
        private static RetrievalCase[] IdentifierCases() =>
        [
            new("E-4022", "err-4022"),
            new("E-5310", "err-5310"),
            new("PL-88-40021", "policy-40021"),
            new("CL-18", "form-cl18"),
            new("what is form CL-17 for?", "form-cl17"),
        ];

        // Natural questions with some lexical overlap — the everyday case, where neither arm is obviously right.
        private static RetrievalCase[] MixedCases() =>
        [
            new("upload fails because the file is too big", "err-4021"),
            new("which policy covers Northwind Logistics?", "policy-40021"),
            new("how many days do I have to report an incident", "claims-howto"),
            new("how long are documents kept", "data-retention"),
            new("who is allowed to drive the vehicles", "drivers"),
        ];

        [LocalOnlyFact]
        public void Hybrid_VsDense_RecallByQueryKind()
        {
            if (!File.Exists(TestModelPaths.MiniLm.SafetensorsPath))
            {
                _out.WriteLine($"missing MiniLM fixture at {TestModelPaths.MiniLm.Dir}");
                return;
            }

            using var embedder = SentenceEmbedder.ForMiniLm(TestModelPaths.MiniLm.Dir);

            // One corpus, indexed once, shared by both retrievers — so the ONLY difference between the two
            // measurements is the retrieval strategy.
            var hybrid = new HybridRetriever(embedder.Dimension, Corpus.Length);
            foreach (var (id, text) in Corpus)
            {
                hybrid.Add(id, embedder.Embed(text), text);
            }

            var denseEvaluator = new RagEvaluator(hybrid.Vectors, embedder.EmbedQuery);
            var hybridEvaluator = RagEvaluator.ForHybrid(hybrid, embedder);

            const int TopK = 5;

            var groups = new (string Name, RetrievalCase[] Cases)[]
            {
                ("semantic", SemanticCases()),
                ("identifier", IdentifierCases()),
                ("mixed", MixedCases()),
            };

            _out.WriteLine($"=== Hybrid vs dense, MiniLM, {Corpus.Length} chunks, recall@{TopK} ===");
            _out.WriteLine($"  {"group",-12} {"dense R@K",10} {"hybrid R@K",11} {"dense MRR",10} {"hybrid MRR",11}");

            var allCases = new List<RetrievalCase>();
            foreach (var group in groups)
            {
                var dense = denseEvaluator.EvaluateRetrieval(group.Cases, TopK);
                var hyb = hybridEvaluator.EvaluateRetrieval(group.Cases, TopK);
                allCases.AddRange(group.Cases);

                _out.WriteLine(
                    $"  {group.Name,-12} {dense.RecallAtK,10:F2} {hyb.RecallAtK,11:F2} "
                    + $"{dense.MeanReciprocalRank,10:F3} {hyb.MeanReciprocalRank,11:F3}");
            }

            var denseAll = denseEvaluator.EvaluateRetrieval(allCases, TopK);
            var hybridAll = hybridEvaluator.EvaluateRetrieval(allCases, TopK);

            _out.WriteLine(
                $"  {"OVERALL",-12} {denseAll.RecallAtK,10:F2} {hybridAll.RecallAtK,11:F2} "
                + $"{denseAll.MeanReciprocalRank,10:F3} {hybridAll.MeanReciprocalRank,11:F3}");

            _out.WriteLine(string.Empty);
            _out.WriteLine("  per-case rank (0 = missed); lexical arm shown where hybrid did not improve:");
            for (var i = 0; i < allCases.Count; i++)
            {
                var denseRank = denseAll.Cases[i].Rank;
                var hybridRank = hybridAll.Cases[i].Rank;
                var worse = hybridRank == 0 || (denseRank > 0 && hybridRank > denseRank);

                _out.WriteLine(
                    $"    dense {denseRank}  hybrid {hybridRank}   \"{allCases[i].Query}\"");

                if (worse)
                {
                    // What the lexical arm dragged in is the whole explanation for a hybrid regression.
                    var lexical = hybrid.Lexical.Search(allCases[i].Query, 3);
                    var ids = new List<string>();
                    for (var j = 0; j < lexical.Length; j++)
                    {
                        ids.Add($"{lexical[j].Id}({lexical[j].Score:F2})");
                    }
                    _out.WriteLine($"        lexical top-3: {string.Join(", ", ids)}");
                }
            }

            // Assert only the claim the measurement actually supports: hybrid must rank identifier lookups
            // at least as well as dense. An assertion on the OVERALL number was tried and removed — it failed,
            // for the reason recorded in the class remarks, and pinning it would have pinned the corpus rather
            // than the retriever.
            var denseIdentifier = denseEvaluator.EvaluateRetrieval(IdentifierCases(), TopK);
            var hybridIdentifier = hybridEvaluator.EvaluateRetrieval(IdentifierCases(), TopK);

            Assert.True(
                hybridIdentifier.MeanReciprocalRank >= denseIdentifier.MeanReciprocalRank,
                $"hybrid identifier MRR {hybridIdentifier.MeanReciprocalRank:F3} fell below dense "
                + $"{denseIdentifier.MeanReciprocalRank:F3} — the one thing the lexical arm is bought for");
        }

        [Fact]
        public void ForHybridEvaluator_RejectsFalsePremiseChecks()
        {
            // RRF scores are rank-derived, so a cosine "grounded" threshold is meaningless against them.
            // Failing loudly beats returning a confident-looking number.
            var retriever = new HybridRetriever(dimension: 2);
            retriever.Add("a", [1f, 0f], "alpha");

            var evaluator = RagEvaluator.ForHybrid(retriever, _ => [1f, 0f]);

            Assert.Throws<OverfitRuntimeException>(
                () => evaluator.EvaluateFalsePremise([new FalsePremiseCase("anything")]));
        }
    }
}
