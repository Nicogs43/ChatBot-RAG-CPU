# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MarineChat: a RAG chatbot for the maritime domain (Italian ports and sailing) built on Small Language Models that run on Intel CPUs through OpenVINO. It was an MSc thesis project; the thesis PDF is in the repo root. The pipeline uses LangChain, a FAISS vector store, BGE embeddings, an optional BGE reranker, and HuggingFace `optimum-intel` models exported to OpenVINO IR.

## Commands

Dependencies are managed with uv: `pyproject.toml` holds the direct dependencies (lower bounds only), `uv.lock` the full resolution. The thesis experiments ran on a 2024 stack (OpenVINO 2024.3, optimum-intel 1.19, transformers 4.44, LangChain 0.2); its exact pins are in commit `f944926`, and `uv sync` creates `.venv` (Python 3.12, see `.python-version` and `pyrightconfig.json`). The project is not an installable package, and has no test suite or build step. Ruff is in the `dev` group but has no config yet.

```bash
uv sync                             # create/update .venv from uv.lock
uv add <pkg> / uv lock --upgrade-package <pkg>   # change dependencies (never edit uv.lock by hand)

# Model download/export (run from repo root; writes to ./model, ./reranker)
uv run python download_model.py            # interactive: HF id + int4/int8/no_compression
uv run python download_reranker_model.py   # -> reranker/bge-reranker-v2-m3_quantized
uv run python download_embedding_model.py  # -> models/ (if missing, the embedder is re-converted from HF at every start: slow)

# bin/ scripts resolve their paths from the repo root (config.ROOT), so they run from any directory
cd bin
uv run python create_vectordb.py [docs...] --splitter_name RecursiveCharacter --chunk_size 400 --chunk_overlap 50 --embedding_model BAAI/bge-small-en-v1.5 --vectorstore_path ../vectorstore_ov_400_50
uv run python main.py [--model phi_int4]          # CLI chat loop, streams the answer (default phi_int4)
uv run python main_gradio.py [--model qwen_0.5b] [--no-browser]  # Gradio UI on http://127.0.0.1:7860 (default config.default_model)
uv run python test_SLMs.py [--models qwen_0.5b ...] [--output-dir DIR]  # answers the 25 eval questions with every (or the given) model -> data/<model>_data_with_answer.csv
# test_SLMs.py and time_inference.py overwrite the thesis results tracked in git unless --output-dir points elsewhere
uv run python time_inference.py --mode first  # default: one cold call on question 1 per model -> ../data/inference_time/*_first_question.csv
uv run python time_inference.py --mode all [--pause 1.0]  # all 25 questions (inference-steps series) -> ../data/inference_time/*_new.csv
# time_inference.py also takes --models and --output-dir; both modes record init time, inference time and RAM (RSS + private bytes, peak during inference)
```

## Architecture

- [bin/config.py](bin/config.py) is the central config: `ROOT`-based paths, the two system prompts, `USER_PROMPT_TEMPLATE`, the `ov_config` OpenVINO runtime settings (u8 KV cache, LATENCY hint), `generation_kwargs`, `retrieval_config` (top_k=5, rerank_top_n=2, threshold=0.6) and `model_dict`. `model_dict` maps a short model key (`phi_int4`, `qwen_1.5b`, …) to its local IR path and system prompt. The evaluation scripts iterate over it, and the key also prefixes the output CSV filenames in `data/`. To add a model, add an entry there and export it with `download_model.py`.
- [bin/vectordb.py](bin/vectordb.py): the embedding loader (`OpenVINOBgeEmbeddings`, local export in `models/` preferred), the reranker loader (`OpenVINOReranker`), PDF loading/splitting, `create_vectordb` (builds and saves FAISS), and `retrieve_documents` (vector search with `similarity_score_threshold`, then the reranker narrows the results; the reranker drops document metadata, so `source`/`page` are copied back and `relevance_score` is added).
- [bin/chatbot.py](bin/chatbot.py): `load_retrieval_resources` (FAISS + reranker, shared by all models) and `RagEngine(model_key, vectorstore, reranker)`. The engine loads the SLM directly with optimum-intel (`OVModelForCausalLM` on CPU, no LangChain wrapper) and exposes `retrieve`, `build_prompt`, `stream` (yields text pieces), `invoke` (returns `{"input", "context", "answer"}`, built on `stream`) and `cancel`. Generation runs in a thread behind a `TextIteratorStreamer` with no timeout, because large models need over a minute before the first token on a slow CPU (LangChain's `HuggingFacePipeline.stream` gives up after 60 s). Closing the `stream` generator early or calling `cancel()` stops generation at the next token.
- Prompts are rendered with each tokenizer's own chat template (`apply_chat_template`), not hand-written markers. Templates without a system role (Gemma) get the system prompt prepended to the user turn. Answers contain only the generated text, not the echoed prompt.
- Every entry point (`main.py`, `main_gradio.py`, `test_SLMs.py`, `time_inference.py`) goes through `load_retrieval_resources` + `RagEngine`; retrieval and generation parameters are changed once, in `config.py`.
- [bin/main_gradio.py](bin/main_gradio.py) is a `gr.Blocks` page (not `ChatInterface`): chat on the left; on the right the passages used (page = PDF index + 1, reranker relevance) and the timings of the last answer. One model is in memory at a time, shared by all visitors; the dropdown lists `config.available_models()` (exports present on disk) and `load_model` frees the old model before loading the new one. Ask turns into Stop while answering; Stop cancels the Gradio event, which closes the `stream` generator and stops the model. Chat history comes back from Gradio with `content` as a list of parts, not a string (see `_message_text`). The look is a nautical-chart palette (white deep water, blue shallow water for the conversation, buff land for document passages, magenta for actions), defined by `theme` and `CSS` with light and dark values.
- The embedding model used to query FAISS must match the one used to build it (`config.embedding_model_id`, default `BAAI/bge-small-en-v1.5`).
- `FAISS.load_local` uses `allow_dangerous_deserialization=True`, because the index is a local pickle.
- Phi and Gemma are slow on the development machine (minutes per question, Gemma ~15 min to load): use `qwen_0.5b` for quick checks.

## Evaluation data flow

`data/questions_ground_truths_25.csv` → `test_SLMs.py` → `data/<model>_data_with_answer.csv` → `proc_data.py` → JSON → RAGAS metrics and plots in the `evaluation/*.ipynb` notebooks (written for Colab). The thesis-era CSVs have the whole echoed Qwen prompt in the `answer` column, generated with the old hand-written templates on the 2024 stack; `proc_data.py` parses the context out of the chat markers for those. New runs write clean answers plus a `contexts` column (JSON list), which `proc_data.py` uses directly. Timing and RAM results live in `data/inference_time/`. Its README is in Italian: the key RAM column is `proc_rss_mb_peak_during_inference`, the `qwen_3b` entry is actually int8, and Phi is never tested unquantized.
