"""Execute the final four-query experiment and save report-ready evidence."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_retrieval_experiment import (
    BI_ENCODER_NAME,
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    CROSS_ENCODER_NAME,
    EXPECTED_SHA256,
    OUTPUT_DIR,
    PDF_PATH,
    chunk_pages,
    dense_top_k,
    extract_pages,
    rerank,
    sha256,
)


LOCAL_LLM_NAME = os.getenv("LOCAL_LLM_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
LATENCY_REPEATS = 5

FINAL_CASES = [
    {
        "case_id": "F1",
        "failure_mode": "Keyword trap / specific rule",
        "query": "What is the failed-try limit for the local PIN that unlocks a multifactor authenticator?",
        "gold_phrase": "no more than 10",
        "expected": "No more than 10 consecutive failed activation attempts.",
    },
    {
        "case_id": "F2",
        "failure_mode": "Negation and modality constraint",
        "query": "Which biometric comparison method is explicitly forbidden even though face, iris, and fingerprints are discussed?",
        "gold_phrase": "Biometric comparison based on voice SHALL NOT be used",
        "expected": "Voice-based biometric comparison is prohibited.",
    },
]


def make_prompt(query: str, selected_chunks: list) -> str:
    context = "\n\n".join(
        f"[{chunk.chunk_id}, pages {chunk.page_start}-{chunk.page_end}]\n{chunk.text}"
        for chunk in selected_chunks
    )
    return f"""Use only the context below to answer the question.
If the context does not contain enough information, say exactly: Insufficient context.
Answer in one sentence of no more than 15 words.

Context:
{context}

Question: {query}
Answer:"""


def generate_answer(prompt: str, tokenizer, model) -> tuple[str, float]:
    messages = [
        {
            "role": "system",
            "content": "You answer questions only from supplied documentary evidence.",
        },
        {"role": "user", "content": prompt},
    ]
    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered = prompt
    inputs = tokenizer(rendered, return_tensors="pt")
    started = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=22,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - started
    answer_tokens = outputs[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip(), elapsed


def median_latencies(query, bi_encoder, cross_encoder, embeddings, chunks):
    baseline_samples = []
    rerank_samples = []
    advanced_samples = []
    for _ in range(LATENCY_REPEATS):
        dense_indices, _, dense_ms = dense_top_k(query, bi_encoder, embeddings)
        _, _, rerank_ms = rerank(query, dense_indices, chunks, cross_encoder)
        baseline_samples.append(dense_ms)
        rerank_samples.append(rerank_ms)
        advanced_samples.append(dense_ms + rerank_ms)
    return {
        "baseline_ms": statistics.median(baseline_samples),
        "reranking_only_ms": statistics.median(rerank_samples),
        "advanced_total_ms": statistics.median(advanced_samples),
    }


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if sha256(PDF_PATH) != EXPECTED_SHA256:
        raise RuntimeError("The PDF checksum does not match the selected assignment corpus")

    pages = extract_pages(PDF_PATH)
    chunks = chunk_pages(pages, CHUNK_WORDS, CHUNK_OVERLAP)
    bi_encoder = SentenceTransformer(BI_ENCODER_NAME, device="cpu")
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device="cpu")
    embeddings = bi_encoder.encode(
        [chunk.text for chunk in chunks],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    # Warm both models before timed trials.
    warm_indices, _, _ = dense_top_k(FINAL_CASES[0]["query"], bi_encoder, embeddings)
    rerank(FINAL_CASES[0]["query"], warm_indices, chunks, cross_encoder)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_NAME)
    llm = AutoModelForCausalLM.from_pretrained(LOCAL_LLM_NAME)

    results = []
    rank_rows = []
    for case in FINAL_CASES:
        query = case["query"]
        dense_indices, dense_scores, _ = dense_top_k(query, bi_encoder, embeddings)
        reranked_indices, cross_scores, _ = rerank(
            query, dense_indices, chunks, cross_encoder
        )
        dense_ranks = {int(index): rank for rank, index in enumerate(dense_indices, 1)}
        reranked_ranks = {
            int(index): rank for rank, index in enumerate(reranked_indices, 1)
        }
        gold_indices = [
            index
            for index, chunk in enumerate(chunks)
            if case["gold_phrase"].lower() in chunk.text.lower()
        ]
        eligible_gold = [index for index in gold_indices if index in dense_ranks]
        if not eligible_gold:
            raise RuntimeError(f"Gold chunk for {case['case_id']} is outside dense Top 10")
        gold_index = min(eligible_gold, key=lambda index: reranked_ranks[index])

        baseline_chunks = [chunks[int(index)] for index in dense_indices[:3]]
        advanced_chunks = [chunks[int(index)] for index in reranked_indices[:3]]
        answer1, answer1_seconds = generate_answer(
            make_prompt(query, baseline_chunks), tokenizer, llm
        )
        answer2, answer2_seconds = generate_answer(
            make_prompt(query, advanced_chunks), tokenizer, llm
        )
        latencies = median_latencies(
            query, bi_encoder, cross_encoder, embeddings, chunks
        )

        for candidate_index in dense_indices:
            candidate_index = int(candidate_index)
            chunk = chunks[candidate_index]
            rank_rows.append(
                {
                    "case_id": case["case_id"],
                    "query": query,
                    "chunk_id": chunk.chunk_id,
                    "pages": f"{chunk.page_start}-{chunk.page_end}",
                    "dense_rank": dense_ranks[candidate_index],
                    "reranked_rank": reranked_ranks[candidate_index],
                    "dense_score": float(
                        dense_scores[dense_ranks[candidate_index] - 1]
                    ),
                    "cross_score": float(
                        cross_scores[reranked_ranks[candidate_index] - 1]
                    ),
                    "is_gold": candidate_index == gold_index,
                    "text": chunk.text,
                }
            )

        results.append(
            {
                **case,
                "gold_chunk_id": chunks[gold_index].chunk_id,
                "gold_pages": f"{chunks[gold_index].page_start}-{chunks[gold_index].page_end}",
                "dense_rank": dense_ranks[gold_index],
                "reranked_rank": reranked_ranks[gold_index],
                "baseline_top3": [chunk.chunk_id for chunk in baseline_chunks],
                "reranked_top3": [chunk.chunk_id for chunk in advanced_chunks],
                "answer1_baseline": answer1,
                "answer2_reranked": answer2,
                "answer1_generation_seconds": answer1_seconds,
                "answer2_generation_seconds": answer2_seconds,
                **latencies,
            }
        )
        print(
            f"{case['case_id']}: {chunks[gold_index].chunk_id} "
            f"rank {dense_ranks[gold_index]} -> {reranked_ranks[gold_index]}"
        )
        print("Answer 1:", answer1)
        print("Answer 2:", answer2)

    pd.DataFrame(rank_rows).to_csv(OUTPUT_DIR / "final_rank_details.csv", index=False)
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "final_results.csv", index=False)
    (OUTPUT_DIR / "final_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
