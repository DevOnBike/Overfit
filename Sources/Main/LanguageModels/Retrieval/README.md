# `LanguageModels/Retrieval` — RAG storage and search

Vector and lexical retrieval, in-process. `VectorStore` and `PersistentVectorStore` hold embedded
chunks; `Bm25Index` is the lexical side; `HybridRetriever` runs both and merges with
`ReciprocalRankFusion`; `TopKMatchSelector` does the selection without sorting the whole candidate set.

Hybrid rather than vector-only because the two fail differently: embeddings miss exact identifiers,
version numbers and rare proper nouns, which is precisely what BM25 is good at, and BM25 misses
paraphrase. Fusing ranks rather than scores avoids having to calibrate two incomparable scales.

Embeddings come from `../Embeddings` (BERT-family sentence encoders, cosine-parity with the reference
implementations on MiniLM, BGE and E5).

## `Evaluation/` — treating retrieval quality as testable

The subdirectory is the part worth knowing about. Retrieval is where a RAG system silently degrades,
so these turn "it seems to work" into assertions:

| Type | Question |
|---|---|
| `RetrievalCase` / `RetrievalReport` | Does the expected chunk come back, at what rank? |
| `ParaphraseGroup` / `ParaphraseStabilityReport` | Do rewordings of one question retrieve the same thing? |
| `FalsePremiseCase` / `FalsePremiseReport` | Does a question with a false premise retrieve *nothing*, rather than the nearest plausible chunk? |
| `CorpusLinter` | Is the corpus itself the problem — duplicates, empty chunks, boilerplate? |
| `RagAssert` | The above as test assertions. |

The false-premise set is the one people skip and the one that catches the worst behaviour: a store
that always returns its nearest neighbour will happily supply confident context for a question about
something that does not exist.
