# Report evidence worksheet

Re-run the notebook after changing any
model, query, chunking, or hardware configuration, then copy the new evidence.

## Reproducibility facts

- Document: NIST SP 800-63B-4, *Digital Identity Guidelines: Authentication and
  Authenticator Management*, July 2025
- Official source: <https://doi.org/10.6028/NIST.SP.800-63B-4>
- PDF SHA-256: `d9ab89576b5ae2cfb3ba37781110788dea27a85a5079496dbfc1a79834670afd`
- Pages extracted: 129
- Shared corpus: 307 chunks
- Chunking: 180 words, 40-word overlap
- Bi-encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Local generator used for the recorded answers:
  `Qwen/Qwen2.5-0.5B-Instruct`

## Verified rank shifts

| Case | Failure type | Gold chunk | Pages | Dense rank | Re-ranked rank |
|---|---|---:|---:|---:|---:|
| F1 | Keyword trap / specific rule | C0129 | 49-49 | 7 | 1 |
| F2 | Negation and modality constraint | C0113 | 43-44 | 4 | 1 |

## Answer evidence

### F1: Local activation PIN retry limit

- Query: “What is the failed-try limit for the local PIN that unlocks a
  multifactor authenticator?”
- Reference answer: No more than 10 consecutive failed activation attempts.
- Baseline Answer 1: “The failed-try limit for the local PIN that unlocks a
  multifactor authenticator is 64 bits.”
- Re-ranked Answer 2: “The failed-try limit for the local PIN that unlocks a
  multifactor authenticator is 10.”
- Observation to explain in your words: the dense Top 3 contained nearby
  authentication-number passages, while the specific activation-secret rule
  was at rank 7. Re-ranking promoted it to rank 1 and corrected the answer.

### F2: Prohibited biometric comparison

- Query: “Which biometric comparison method is explicitly forbidden even
  though face, iris, and fingerprints are discussed?”
- Reference answer: Voice-based biometric comparison is prohibited.
- Baseline Answer 1: “Biometric comparison methods are explicitly forbidden
  despite face, iris, and fingerprints being discussed.”
- Re-ranked Answer 2: “Face, iris, and fingerprints are discussed, but facial
  images cannot be taken directly.”
- Observation to explain in your words: re-ranking successfully moved the
  exact voice-prohibition passage from rank 4 to rank 1, but the 0.5B local
  generator still failed to use it. This is a retrieval success and a
  generator-following failure, not an end-to-end answer success.

For a stronger final comparison, set `GENERATOR_BACKEND=gemini`, provide
`GEMINI_API_KEY` through the environment, re-run the notebook, and retain the
raw Answer 1 and Answer 2 outputs. Do not paste API keys into the notebook.

## Latency evidence

The notebook reports the median of five warmed CPU runs per query. It excludes
PDF extraction, model loading, one-time corpus embedding, and LLM generation.
Treat the exact values as machine- and load-dependent and copy them from
`outputs/notebook_latency.csv` after the final run.

In the latest verified run:

| Case | Baseline retrieval | Re-ranking only | Advanced total |
|---|---:|---:|---:|
| F1 | 54.74 ms | 256.48 ms | 397.90 ms |
| F2 | 15.21 ms | 1251.41 ms | 1266.62 ms |

## Prompts for the four required report sections

Write short answers to these prompts yourself:

1. Architecture comparison
   - Why can chunk embeddings be precomputed for a bi-encoder?
   - What additional query-passage interaction does a cross-encoder obtain?

2. Qualitative and rank-shift analysis
   - Why did F1 confuse bit length with a retry count?
   - What changed when C0129 moved from rank 7 to rank 1?
   - Why is F2 evidence that retrieval quality and generation quality are
     separate concerns?

3. Latency profiling
   - State the hardware/backend and the timing boundary.
   - Compare baseline retrieval with the full advanced retrieval path.

4. Scaling
   - Explain why applying a cross-encoder to every chunk requires one expensive
     forward pass per query-chunk pair.
   - Explain why dense retrieval is used to reduce the candidate set first.

## Remaining acceptance check

Before submission, obtain at least two cases where the assignment's desired
claim is defensible. The retrieval requirement is already met for F1 and F2.
Only F1 currently demonstrates an end-to-end corrected local-model answer. A
stronger generator run should be completed before claiming that both Answer 2
outputs improved.
