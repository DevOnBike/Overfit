// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Retrieval;

namespace DevOnBike.Overfit.Tests.LanguageModels.Retrieval
{
    /// <summary>
    /// Pins the BM25 lexical arm: the ranking behaviour that makes it worth having next to a
    /// <see cref="VectorStore"/> (exact-term retrieval, IDF, length normalisation) and the tokenisation it
    /// shares between indexing and querying.
    /// </summary>
    public sealed class Bm25IndexTests
    {
        [Fact]
        public void Search_FindsDocumentContainingTheExactTerm()
        {
            var index = new Bm25Index();
            index.Add("a", "the cat sat on the mat");
            index.Add("b", "a dog barked loudly");
            index.Add("c", "birds are singing");

            var hits = index.Search("dog", 3);

            Assert.Single(hits);
            Assert.Equal("b", hits[0].Id);
        }

        [Fact]
        public void Search_DoesNotReturnDocumentsWithoutAnyQueryTerm()
        {
            var index = new Bm25Index();
            index.Add("a", "alpha beta");
            index.Add("b", "gamma delta");

            // Asking for 10 must not pad the result with zero-scoring documents.
            var hits = index.Search("alpha", 10);

            Assert.Single(hits);
            Assert.Equal("a", hits[0].Id);
        }

        [Fact]
        public void Search_UnknownTerm_ReturnsNothing()
        {
            var index = new Bm25Index();
            index.Add("a", "alpha beta");

            Assert.Empty(index.Search("omega", 5));
        }

        [Fact]
        public void Search_RareTermOutranksCommonTerm()
        {
            var index = new Bm25Index();
            // "common" appears everywhere (low IDF); "rare" appears once (high IDF).
            index.Add("a", "common word here");
            index.Add("b", "common word there");
            index.Add("c", "common rare word");
            index.Add("d", "common word everywhere");

            var hits = index.Search("common rare", 4);

            Assert.Equal("c", hits[0].Id);
        }

        [Fact]
        public void Search_ShorterDocumentOutranksLongerOne_ForTheSameTermCount()
        {
            var index = new Bm25Index();
            index.Add("short", "needle");
            index.Add("long", "needle " + string.Join(' ', Enumerable.Repeat("filler", 200)));

            var hits = index.Search("needle", 2);

            // Both contain "needle" exactly once; length normalisation (b > 0) must favour the short one.
            Assert.Equal(2, hits.Length);
            Assert.Equal("short", hits[0].Id);
        }

        [Fact]
        public void Search_LengthNormalisationDisabled_TreatsBothLengthsAlike()
        {
            var index = new Bm25Index(b: 0f);
            index.Add("short", "needle");
            index.Add("long", "needle " + string.Join(' ', Enumerable.Repeat("filler", 200)));

            var hits = index.Search("needle", 2);

            // With b = 0 the two documents differ only by length, which is now ignored → identical scores.
            Assert.Equal(hits[0].Score, hits[1].Score, 5);
        }

        [Fact]
        public void Search_RepeatedQueryTerm_DoesNotDoubleCount()
        {
            var index = new Bm25Index();
            index.Add("a", "alpha beta gamma");
            index.Add("b", "alpha delta");

            var once = index.Search("alpha", 2);
            var twice = index.Search("alpha alpha alpha", 2);

            Assert.Equal(once.Length, twice.Length);
            for (var i = 0; i < once.Length; i++)
            {
                Assert.Equal(once[i].Id, twice[i].Id);
                Assert.Equal(once[i].Score, twice[i].Score, 5);
            }
        }

        [Fact]
        public void Search_CarriesThePayload()
        {
            var index = new Bm25Index();
            index.Add("a", "alpha beta", payload: "the original text");

            var hits = index.Search("beta", 1);

            Assert.Equal("the original text", hits[0].Payload);
        }

        [Fact]
        public void Search_EmptyIndexOrEmptyQuery_ReturnsNothing()
        {
            var empty = new Bm25Index();
            Assert.Empty(empty.Search("anything", 5));

            var index = new Bm25Index();
            index.Add("a", "alpha");
            Assert.Empty(index.Search("   ...   ", 5));
        }

        [Fact]
        public void Search_SpanOverload_ReportsHowManyItWrote()
        {
            var index = new Bm25Index();
            index.Add("a", "alpha");
            index.Add("b", "beta");

            Span<VectorMatch> buffer = new VectorMatch[5];
            var written = index.Search("alpha", buffer);

            Assert.Equal(1, written);
            Assert.Equal("a", buffer[0].Id);
        }

        [Fact]
        public void Tokenize_LowercasesAndSplitsOnPunctuation()
        {
            Assert.Equal(["hello", "world", "42"], Bm25Index.Tokenize("Hello, WORLD! (42)"));
        }

        [Fact]
        public void Tokenize_PreservesPolishDiacritics()
        {
            // An ASCII-range split would shred these into fragments and silently wreck recall on Polish text.
            Assert.Equal(["zażółć", "gęślą", "jaźń"], Bm25Index.Tokenize("Zażółć gęślą jaźń!"));
        }

        [Fact]
        public void Tokenize_EmitsIdentifierPartsAndTheJoinedForm()
        {
            // Parts keep component queries working; the joined form carries the IDF that common parts cannot.
            Assert.Equal(["pl", "88", "40021", "pl-88-40021"], Bm25Index.Tokenize("PL-88-40021"));
            Assert.Equal(
                ["overfit", "decode", "workers", "overfit_decode_workers"],
                Bm25Index.Tokenize("OVERFIT_DECODE_WORKERS"));
        }

        [Fact]
        public void Tokenize_DoesNotEmitAJoinedFormForOrdinaryWords()
        {
            // Emitting a duplicate for single-part words would double their term frequency and corrupt both
            // TF saturation and length normalisation.
            Assert.Equal(["hello", "world"], Bm25Index.Tokenize("hello world"));
        }

        [Fact]
        public void Tokenize_ConnectorMustSitBetweenTwoAlphanumerics()
        {
            // Trailing/leading connectors and dashes used as punctuation must not glue terms together.
            Assert.Equal(["abc"], Bm25Index.Tokenize("abc-"));
            Assert.Equal(["abc"], Bm25Index.Tokenize("-abc"));
            Assert.Equal(["one", "two"], Bm25Index.Tokenize("one -- two"));
        }

        [Fact]
        public void Search_FindsAnIdentifierWhosePartsAreAllCommonWords()
        {
            // The regression this tokenisation change was built for: every part is common, so only the joined
            // form can rank the defining document first.
            var index = new Bm25Index();
            index.Add("defines", "Set OVERFIT_DECODE_WORKERS to cap the decode worker count.");
            index.Add("noise-1", "The decode path spawns workers for each overfit projection.");
            index.Add("noise-2", "Overfit workers decode tokens; decode workers are capped.");
            index.Add("noise-3", "Workers decode. Overfit decode workers overfit decode.");

            var hits = index.Search("OVERFIT_DECODE_WORKERS", 4);

            Assert.Equal("defines", hits[0].Id);
        }

        [Fact]
        public void Tokenize_EmptyAndSymbolOnlyInput_YieldsNoTerms()
        {
            Assert.Empty(Bm25Index.Tokenize(string.Empty));
            Assert.Empty(Bm25Index.Tokenize("--- !!! ---"));
        }

        [Fact]
        public void CountsReflectTheIndexedCorpus()
        {
            var index = new Bm25Index();
            index.Add("a", "alpha beta");
            index.Add("b", "beta gamma");

            Assert.Equal(2, index.Count);
            Assert.Equal(3, index.TermCount); // alpha, beta, gamma
        }

        [Fact]
        public void Constructor_RejectsOutOfRangeB()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new Bm25Index(b: -0.1f));
            Assert.Throws<ArgumentOutOfRangeException>(() => new Bm25Index(b: 1.1f));
        }
    }
}
