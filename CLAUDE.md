# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MarineChat: a RAG chatbot for the maritime domain (Italian ports and sailing) built on Small Language Models that run on Intel CPUs through OpenVINO. It was an MSc thesis project; the thesis PDF is in the repo root. The pipeline uses LangChain, a FAISS vector store, BGE embeddings, an optional BGE reranker, and HuggingFace `optimum-intel` models exported to OpenVINO IR.

## Commands

Dependencies are managed with uv: `pyproject.toml` holds the direct dependencies (pinned to the thesis-era versions), `uv.lock` the full resolution, and `uv sync` creates `.venv` (Python 3.12, see `.python-version` and `pyrightconfig.json`). The project is not an installable package, and has no test suite or build step. Ruff is in the `dev` group but has no config yet.

```bash
uv sync                             # create/update .venv from uv.lock
uv add <pkg> / uv lock --upgrade-package <pkg>   # change dependencies (never edit uv.lock by hand)

# Model download/export (run from repo root; writes to ./model, ./reranker)
uv run python download_model.py            # interactive: HF id + int4/int8/no_compression
uv run python download_reranker_model.py   # -> reranker/bge-reranker-v2-m3_quantized
uv run python download_embedding_model.py  # -> models/ (note: runtime code loads the embedder by HF id, not from this folder)

# Everything in bin/ must be run with bin/ as CWD (config paths are "../model/...", "../data/...")
cd bin
uv run python create_vectordb.py [docs...] --splitter_name RecursiveCharacter --chunk_size 400 --chunk_overlap 50 --embedding_model BAAI/bge-small-en-v1.5 --vectorstore_path ../vectorstore_ov_400_50
uv run python main.py          # CLI chat loop (Phi-3.5 int4 hardcoded)
uv run python main_gradio.py   # Gradio UI (default Qwen2.5-0.5B from chatbot.py)
uv run python test_SLMs.py     # answers the 25 eval questions with every model in model_dict -> ../data/<model>_data_with_answer.csv
uv run python time_inference.py --mode first  # default: one cold call on question 1 per model -> ../data/inference_time/*_first_question.csv
uv run python time_inference.py --mode all [--pause 1.0]  # all 25 questions (inference-steps series) -> ../data/inference_time/*_new.csv
# both modes record init time, inference time and RAM (RSS + private bytes, peak during inference)
```

## Architecture

- [bin/config.py](bin/config.py) is the central config: per-model-family prompt templates (Phi `<|system|>`, Qwen ChatML `<|im_start|>`, Gemma `<start_of_turn>`), the `ov_config` OpenVINO runtime settings (u8 KV cache, LATENCY hint), default paths, and `model_dict`. `model_dict` maps a short model key (`phi_int4`, `qwen_1.5b`, …) to its local IR path, prompt template and system prompt. The evaluation scripts iterate over it, and the key also prefixes the output CSV filenames in `data/`. To add a model, add an entry there and export it with `download_model.py`.
- [bin/vectordb.py](bin/vectordb.py): embedding loaders (OpenVINO `OpenVINOBgeEmbeddings` and plain HF), the reranker loader (`OpenVINOReranker`), PDF loading/splitting, `create_vectordb` (builds and saves FAISS), and `create_rag_chain`. That last one builds the retriever (`similarity_score_threshold` with top_k, optionally wrapped in `ContextualCompressionRetriever` with the reranker), then a stuff-documents chain, then `create_retrieval_chain`. Chains are invoked with `{"input": question}` and return `output["answer"]`.
- [bin/chatbot.py](bin/chatbot.py): `initialize_openvino_pipeline` (a `HuggingFacePipeline` with `backend="openvino"` on CPU) and `request_cancel`. The entry points call `request_cancel` after each generation.
- Each entry point (`main.py`, `main_gradio.py`, `test_SLMs.py`, `time_inference.py`) repeats the same setup by hand: load FAISS → load reranker → init the OV pipeline → override `ov_slm.pipeline_kwargs` → `create_rag_chain` (top_k=5, rerank_top_n=2, threshold=0.6). A change to the retrieval or generation parameters has to be made in each file.
- The embedding model used to query FAISS must match the one used to build it (default `BAAI/bge-small-en-v1.5`). `main.py` loads it through OpenVINO and `main_gradio.py` through plain HF.
- `FAISS.load_local` uses `allow_dangerous_deserialization=True`, because the index is a local pickle.

## Evaluation data flow

`data/questions_ground_truths_25.csv` → `test_SLMs.py` → `data/<model>_data_with_answer.csv` → `proc_data.py` (strips Qwen chat markers and separates out the context) → JSON → RAGAS metrics and plots in the `evaluation/*.ipynb` notebooks. Timing and RAM results live in `data/inference_time/`. Its README is in Italian: the key RAM column is `proc_rss_mb_peak_during_inference`, the `qwen_3b` entry is actually int8, and Phi is never tested unquantized.
