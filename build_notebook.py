"""Create the assignment notebook from the verified implementation."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks" / "nist_rag_reranking.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    }

    nb["cells"] = [
        md(
            """# NIST SP 800-63B-4 RAG: Baseline vs. Cross-Encoder Re-Ranking

This notebook compares two workflows over the **same 307 chunks** from one official PDF:

1. Bi-encoder retrieval: dense Top 10, with the first 3 sent to the generator.
2. Advanced retrieval: cross-encode those same 10 candidates and send the re-ranked Top 3.

The notebook records the PDF checksum, rank shifts, paired answers, and warmed CPU latency. The final report must be written in the student's own words; these outputs are experimental evidence, not submission prose."""
        ),
        md(
            """## Environment

Run from the `rag-assignment` directory. Uncomment the installation command in a fresh environment. The default `recorded` backend displays verified local outputs. Select `local` to regenerate them on CPU, or `gemini` after setting `GEMINI_API_KEY`."""
        ),
        code(
            """# %pip install -q pypdf sentence-transformers transformers torch pandas numpy matplotlib requests google-genai

from pathlib import Path
import json, os, statistics, sys, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

GENERATOR_BACKEND = os.getenv("GENERATOR_BACKEND", "recorded")  # recorded | local | gemini
os.environ.setdefault("LOCAL_LLM_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
LATENCY_REPEATS = 5"""
        ),
        md(
            """## 1. Corpus preparation

The helper functions below are ordinary Python code included in this repository. They download/check the official PDF, normalize extracted text, form 180-word windows with 40-word overlap, and preserve page metadata. Both pipelines receive exactly the same chunks."""
        ),
        code(
            """from run_retrieval_experiment import (
    BI_ENCODER_NAME, CROSS_ENCODER_NAME, EXPECTED_SHA256, PDF_PATH,
    CHUNK_WORDS, CHUNK_OVERLAP, chunk_pages, extract_pages, sha256,
    dense_top_k, rerank,
)

assert PDF_PATH.exists(), "Run the download/setup step in README.md first."
assert sha256(PDF_PATH) == EXPECTED_SHA256
pages = extract_pages(PDF_PATH)
chunks = chunk_pages(pages, CHUNK_WORDS, CHUNK_OVERLAP)
print({
    "pdf": PDF_PATH.name,
    "sha256": EXPECTED_SHA256,
    "pages": len(pages),
    "chunks": len(chunks),
    "chunk_words": CHUNK_WORDS,
    "overlap_words": CHUNK_OVERLAP,
})
pd.DataFrame([vars(chunk) for chunk in chunks]).head(3)"""
        ),
        md(
            """## 2. Bi-encoder index and cross-encoder

The bi-encoder embeds queries and chunks independently. Normalized vectors allow cosine similarity through a dot product. The cross-encoder processes each `(query, chunk)` pair jointly, but only for the dense Top 10."""
        ),
        code(
            """from sentence_transformers import CrossEncoder, SentenceTransformer

bi_encoder = SentenceTransformer(BI_ENCODER_NAME, device="cpu")
cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device="cpu")

started = time.perf_counter()
corpus_embeddings = bi_encoder.encode(
    [chunk.text for chunk in chunks], batch_size=32,
    normalize_embeddings=True, convert_to_numpy=True,
    show_progress_bar=True,
)
index_seconds = time.perf_counter() - started
print(f"One-time corpus embedding: {index_seconds:.2f} seconds")"""
        ),
        md(
            """## 3. Stress-test queries

Each selected answer-bearing chunk initially lies at dense rank 4-10 and must move into the re-ranked Top 3. This directly tests the assignment's failure condition."""
        ),
        code(
            """CASES = [
    {
        "case": "F1",
        "mode": "Keyword trap / specific rule",
        "query": "What is the failed-try limit for the local PIN that unlocks a multifactor authenticator?",
        "gold_phrase": "no more than 10",
        "reference": "No more than 10 consecutive failed activation attempts.",
    },
    {
        "case": "F2",
        "mode": "Negation and modality constraint",
        "query": "Which biometric comparison method is explicitly forbidden even though face, iris, and fingerprints are discussed?",
        "gold_phrase": "Biometric comparison based on voice SHALL NOT be used",
        "reference": "Voice-based biometric comparison is prohibited.",
    },
]"""
        ),
        code(
            """runs, candidate_rows = [], []
for case in CASES:
    dense_i, dense_s, _ = dense_top_k(
        case["query"], bi_encoder, corpus_embeddings, top_k=10
    )
    rerank_i, cross_s, _ = rerank(
        case["query"], dense_i, chunks, cross_encoder
    )
    dense_rank = {int(i): r for r, i in enumerate(dense_i, 1)}
    rerank_rank = {int(i): r for r, i in enumerate(rerank_i, 1)}
    gold = [i for i, c in enumerate(chunks)
            if case["gold_phrase"].lower() in c.text.lower() and i in dense_rank]
    assert gold, f"Gold evidence for {case['case']} was not in dense Top 10"
    gold_i = min(gold, key=lambda i: rerank_rank[i])
    assert dense_rank[gold_i] > 3 and rerank_rank[gold_i] <= 3

    runs.append({**case, "dense_i": dense_i, "rerank_i": rerank_i,
                 "gold_i": gold_i, "dense_rank": dense_rank[gold_i],
                 "reranked_rank": rerank_rank[gold_i]})
    for i in dense_i:
        i = int(i)
        candidate_rows.append({
            "case": case["case"], "chunk": chunks[i].chunk_id,
            "pages": f"{chunks[i].page_start}-{chunks[i].page_end}",
            "dense_rank": dense_rank[i], "reranked_rank": rerank_rank[i],
            "gold": i == gold_i, "preview": chunks[i].text[:160],
        })

rank_shifts = pd.DataFrame([{
    "case": r["case"], "failure_mode": r["mode"],
    "gold_chunk": chunks[r["gold_i"]].chunk_id,
    "pages": f"{chunks[r['gold_i']].page_start}-{chunks[r['gold_i']].page_end}",
    "dense_rank": r["dense_rank"], "reranked_rank": r["reranked_rank"],
} for r in runs])
rank_shifts"""
        ),
        code(
            """pd.set_option("display.max_colwidth", 90)
pd.DataFrame(candidate_rows).sort_values(["case", "dense_rank"])"""
        ),
        md(
            """## 4. Identical prompting and paired answers

Only the selected Top 3 chunks change. The recorded run used `Qwen/Qwen2.5-0.5B-Instruct` locally. It demonstrates one successful answer repair and one important limitation: reranking can restore evidence while a weak generator still fails to use it."""
        ),
        code(
            """from run_final_experiment import LOCAL_LLM_NAME, generate_answer, make_prompt

RECORDED_LOCAL_ANSWERS = {
    "F1": {
        "answer1": "The failed-try limit for the local PIN that unlocks a multifactor authenticator is 64 bits.",
        "answer2": "The failed-try limit for the local PIN that unlocks a multifactor authenticator is 10.",
    },
    "F2": {
        "answer1": "Biometric comparison methods are explicitly forbidden despite face, iris, and fingerprints being discussed.",
        "answer2": "Face, iris, and fingerprints are discussed, but facial images cannot be taken directly.",
    },
}

if GENERATOR_BACKEND == "local":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_LLM_NAME)
    answers = {}
    for run in runs:
        a1, _ = generate_answer(
            make_prompt(run["query"], [chunks[int(i)] for i in run["dense_i"][:3]]),
            tokenizer, model,
        )
        a2, _ = generate_answer(
            make_prompt(run["query"], [chunks[int(i)] for i in run["rerank_i"][:3]]),
            tokenizer, model,
        )
        answers[run["case"]] = {"answer1": a1, "answer2": a2}
elif GENERATOR_BACKEND == "gemini":
    from google import genai
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY before using the Gemini backend")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    answers = {}
    for run in runs:
        prompt1 = make_prompt(run["query"], [chunks[int(i)] for i in run["dense_i"][:3]])
        prompt2 = make_prompt(run["query"], [chunks[int(i)] for i in run["rerank_i"][:3]])
        a1 = client.models.generate_content(model="gemini-2.5-flash", contents=prompt1).text.strip()
        a2 = client.models.generate_content(model="gemini-2.5-flash", contents=prompt2).text.strip()
        answers[run["case"]] = {"answer1": a1, "answer2": a2}
else:
    answers = RECORDED_LOCAL_ANSWERS

answer_table = pd.DataFrame([{
    "case": r["case"], "reference": r["reference"],
    "answer1_baseline": answers[r["case"]]["answer1"],
    "answer2_reranked": answers[r["case"]]["answer2"],
} for r in runs])
answer_table"""
        ),
        md(
            """### Candid interpretation

- **F1:** Answer 1 confused a bit-length passage with an attempt count. After chunk C0129 moved from rank 7 to rank 1, Answer 2 correctly returned 10.
- **F2:** Chunk C0113 moved from rank 4 to rank 1, so retrieval relevance was restored. The 0.5B generator still failed to state that voice comparison is prohibited. This is a generator-following failure and must not be reported as an end-to-end answer success. Re-run with a stronger API model if available."""
        ),
        md("## 5. Latency profiling"),
        code(
            """# Excludes model loading, PDF extraction, one-time indexing, and generator latency.
latency_rows = []
for case in CASES:
    # Warm up each query shape before collecting its repeated measurements.
    warm_i, _, _ = dense_top_k(case["query"], bi_encoder, corpus_embeddings)
    rerank(case["query"], warm_i, chunks, cross_encoder)
    baseline, reranking, advanced = [], [], []
    for _ in range(LATENCY_REPEATS):
        dense_i, _, dense_ms = dense_top_k(case["query"], bi_encoder, corpus_embeddings)
        _, _, rerank_ms = rerank(case["query"], dense_i, chunks, cross_encoder)
        baseline.append(dense_ms)
        reranking.append(rerank_ms)
        advanced.append(dense_ms + rerank_ms)
    latency_rows.append({
        "case": case["case"],
        "baseline_retrieval_ms": statistics.median(baseline),
        "reranking_only_ms": statistics.median(reranking),
        "advanced_total_ms": statistics.median(advanced),
    })
latency_table = pd.DataFrame(latency_rows)
latency_table"""
        ),
        code(
            """ax = latency_table.set_index("case")[["baseline_retrieval_ms", "advanced_total_ms"]].plot(
    kind="bar", figsize=(7, 4), rot=0
)
ax.set_ylabel("Median latency (ms)")
ax.set_title("Baseline vs. dense retrieval plus Top-10 reranking")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()"""
        ),
        md(
            """## 6. Architecture comparison and scaling

- **Bi-encoder:** query and chunk are embedded independently. Chunk embeddings are reusable, making vector search fast, but there is no direct token-level interaction between query and passage.
- **Cross-encoder:** query and passage are processed jointly with self-attention. It can better resolve negation and fine-grained constraints, but it performs a separate model forward pass for every pair.
- **Why not cross-encode everything?** For `N` chunks, whole-corpus cross-encoding requires `N` expensive forward passes for every query and cannot reuse precomputed chunk scores. Here dense retrieval reduces 307 chunks to 10 before reranking; the same saving is essential at larger scale."""
        ),
        md("## 7. Export evidence for the student-written report"),
        code(
            """OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
rank_shifts.to_csv(OUTPUT_DIR / "notebook_rank_shifts.csv", index=False)
answer_table.to_csv(OUTPUT_DIR / "notebook_answers.csv", index=False)
latency_table.to_csv(OUTPUT_DIR / "notebook_latency.csv", index=False)
metadata = {
    "pdf_sha256": EXPECTED_SHA256, "pages": len(pages), "chunks": len(chunks),
    "chunk_words": CHUNK_WORDS, "chunk_overlap": CHUNK_OVERLAP,
    "bi_encoder": BI_ENCODER_NAME, "cross_encoder": CROSS_ENCODER_NAME,
    "generator_backend": GENERATOR_BACKEND,
    "local_generator": LOCAL_LLM_NAME, "index_seconds": index_seconds,
}
(OUTPUT_DIR / "notebook_metadata.json").write_text(json.dumps(metadata, indent=2))
metadata"""
        ),
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
