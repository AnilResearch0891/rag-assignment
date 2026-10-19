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
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
mkdir -p data
curl -L --fail \
  https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63B-4.pdf \
  -o data/NIST.SP.800-63B-4.pdf
shasum -a 256 data/NIST.SP.800-63B-4.pdf
```

Use Python 3.11 for this environment. The pinned NumPy 2.2.6 requires Python
3.10 or newer. The standard
PyTorch package installed from PyPI uses CPU on macOS unless supported Apple
GPU acceleration is selected automatically.

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
CPU/GPU time. The notebook and final experiment script support Gemini and
OpenAI API generation. Keep API keys in environment variables, never in the
notebook or repository.

For Gemini, use `GENERATOR_BACKEND=gemini` and `GEMINI_API_KEY`. For OpenAI,
use `GENERATOR_BACKEND=openai` and `OPENAI_API_KEY`. The OpenAI model defaults
to `gpt-6-luna`; override it with `OPENAI_MODEL` if needed. For example:

```bash
export GENERATOR_BACKEND=gemini
export GEMINI_API_KEY="your-key"
.venv/bin/python run_final_experiment.py
```

For OpenAI, change the backend and supply the OpenAI API key in your shell.
Then run:

```bash
export GENERATOR_BACKEND=openai
export OPENAI_API_KEY="your-key"
OPENAI_MODEL=gpt-6-luna .venv/bin/python run_final_experiment.py
```

The key is read only from the environment. The runner uses `gemini-3.8-flash`
by default when using Gemini; set `GEMINI_MODEL` to choose another compatible
model.

To let the runner load a local API key without putting it in shell history,
copy `.env.example` to `.env`, replace the placeholder with your Gemini key,
and keep the file private. `.env` is ignored by Git. The final runner loads
this file automatically.

## Reproducibility boundary

The PDF and model weights are downloaded from their official upstream sources
and cached locally. Generated outputs and downloaded binaries are excluded from
Git. The notebook records source identifiers, model names, configuration,
random seed, and the PDF checksum.

Do not treat the included candidate queries as proven failure cases until the
executed rank tables show that the required chunk begins below baseline rank 3
and is promoted into the re-ranked Top 3.
