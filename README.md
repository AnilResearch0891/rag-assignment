# RAG Assignment: Baseline Retrieval vs. Cross-Encoder Re-Ranking

This directory contains the reproducible assignment implementation using
**NIST SP 800-63B-4, Digital Identity Guidelines: Authentication and
Authenticator Management**.

The primary deliverable is `notebooks/nist_rag_reranking.ipynb`. It downloads
the official PDF, verifies the expected file hash, builds one shared chunk
corpus, compares dense Top-3 retrieval against cross-encoder re-ranking of the
dense Top-10, profiles latency, and exports report-ready evidence.

## Setup

```bash
cd rag-assignment
python -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
.venv/bin/pip install -r requirements.txt
mkdir -p data
curl -L --fail \
  https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63B-4.pdf \
  -o data/NIST.SP.800-63B-4.pdf
sha256sum data/NIST.SP.800-63B-4.pdf
```

The separate CPU PyTorch installation avoids downloading CUDA libraries on a
machine without a GPU.

The expected PDF SHA-256 is
`d9ab89576b5ae2cfb3ba37781110788dea27a85a5079496dbfc1a79834670afd`.

## Testing other generators

The executed notebook defaults to the recorded Qwen 0.5B outputs so it opens
quickly. To run a different Hugging Face causal/instruction model, set the
backend and model name before starting Jupyter:

```bash
export GENERATOR_BACKEND=local
export LOCAL_LLM_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

Other `AutoModelForCausalLM` models can be substituted in the same way. Larger
models generally follow retrieved evidence better but require more memory and
CPU/GPU time. The notebook also supports `GENERATOR_BACKEND=gemini` with a
`GEMINI_API_KEY` supplied through the environment. Never store API keys in the
notebook or repository.

## Reproducibility boundary

The PDF and model weights are downloaded from their official upstream sources
and cached locally. Generated outputs and downloaded binaries are excluded from
Git. The notebook records source identifiers, model names, configuration,
random seed, and the PDF checksum.

Do not treat the included candidate queries as proven failure cases until the
executed rank tables show that the required chunk begins below baseline rank 3
and is promoted into the re-ranked Top 3.
