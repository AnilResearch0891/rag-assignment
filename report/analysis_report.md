# Failure Mode Analysis and Retrieval Report

## 3. Failure mode analysis (stress testing)

The experiment used NIST SP 800-63B-4, divided into 307 overlapping chunks. For each query, dense retrieval selected ten candidates; the cross-encoder reordered those candidates, and each pipeline supplied its top three chunks to Gemini Flash-Lite. Both cases below expose a specific weakness in dense retrieval, and reranking moved the gold passage to the top.

| Case | Failure mode and query | Gold passage rank | Baseline Answer 1 | Re-ranked Answer 2 |
|---|---|---:|---|---|
| F1 | **Keyword trap / specific rule.** “What is the failed-try limit for the local PIN that unlocks a multifactor authenticator?” The document calls the numeric activation secret a PIN; the exact retry rule is phrased in terms of activation attempts. | C0129, p. 49: **#7 → #1** | “Insufficient context” | “The authenticator or verifier SHALL implement a retry-limiting mechanism that limits the number of consecutive failed activation attempts using the authenticator to no more than 10.” |
| F2 | **Negation and modality.** “Which biometric comparison method is explicitly forbidden even though face, iris, and fingerprints are discussed?” | C0113, pp. 43–44: **#4 → #1** | “Insufficient context” | “Biometric comparison based on voice SHALL NOT be used.” |

In both baseline cases, Gemini omitted the answer because the relevant clause was outside the dense Top 3. After reranking promoted the exact passages, Gemini recovered the ten-attempt limit and the voice-comparison prohibition. Neither baseline answer hallucinated a rule in this run; both abstained. The generator was Gemini `gemini-3.5-flash-lite`, and the answer prompt directs it to preserve SHALL/SHALL NOT wording and distinguish activation attempts from general authentication attempts.

## 4. Analysis and report

### Architecture comparison

A bi-encoder embeds each passage independently and embeds the query separately. The passage vectors can be precomputed once; retrieval compares the query vector with all passage vectors using cosine similarity. A cross-encoder takes a query and passage together, allowing token-level attention between them and a more specific relevance judgment. That joint comparison costs a model inference for every candidate passage, so it is used after the bi-encoder narrows the search.

### Qualitative and rank-shift analysis

F1 is a document-specific terminology trap: the query says “local PIN,” while the decisive rule refers to “activation secret” and “failed activation attempts.” Dense retrieval’s Top 3 did not include that rule. C0129 moved from rank 7 to rank 1, and the answer changed from abstention to the correct limit of no more than 10 consecutive failures. F2 tests a negative constraint: broad discussion of biometric modalities obscured the explicit SHALL NOT prohibition. C0113 moved from rank 4 to rank 1, allowing the answer to identify voice comparison and retain the prohibition’s normative force.

### Latency profiling

The following are medians of five warmed CPU retrieval trials per query from the current run. Baseline time includes query embedding and dense ranking. Re-ranking time covers scoring the ten dense candidates; advanced total is the median of the per-trial baseline-plus-reranking times. These figures exclude document extraction, model loading, one-time corpus embedding, and answer generation.

| Case | Baseline retrieval | Re-ranking only (10 candidates) | Advanced retrieval total |
|---|---:|---:|---:|
| F1 | 12.44 ms | 268.43 ms | 280.73 ms |
| F2 | 12.26 ms | 215.94 ms | 231.20 ms |

On this CPU run, adding the cross-encoder substantially increased retrieval latency, while enabling the two critical passages to reach rank 1. The exact values depend on hardware and system load.

### Why not cross-encode the full collection?

The cross-encoder must process the query jointly with each passage. Scoring all 307 chunks would require 307 query-passage inferences per query, with token-level attention over each pair. The bi-encoder can precompute passage vectors and quickly score the whole corpus; applying the cross-encoder only to the ten strongest candidates keeps the expensive pairwise work bounded while repairing ranking errors among likely matches.

## Reproducibility

- Corpus: NIST SP 800-63B-4 (July 2025), 129 extracted pages, 307 chunks; 180 words per chunk with 40-word overlap.
- Bi-encoder: `sentence-transformers/all-MiniLM-L6-v2`.
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Generator: Gemini `gemini-3.5-flash-lite` through the Gemini API.
- Raw results: `outputs/final_results.csv` and `outputs/final_rank_details.csv`.
