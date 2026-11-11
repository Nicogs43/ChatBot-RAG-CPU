# Navigating the Seas of AI: Deploying Small Language Models on Compact Edge Computers for Maritime Applications 🌊⛵

This repository contains the code for creating **MarineChat** a chatbot based on a Small Language Model (SLM) integrated with Retrieval-Augmented Generation (R.A.G.), optimized to run on an Intel CPU using the [OpenVINO toolkit](https://github.com/openvinotoolkit/openvino).

Check [here](https://docs.openvino.ai/2024/about-openvino/release-notes-openvino/system-requirements.html) to see if your device supports OpenVINO.

This project was developed as my final work for the MSc in Computer Science, Artificial Intelligence track, at Università di Genova.

---

## Getting Started 🚀

### 1. Install Dependencies
The project is managed with [uv](https://docs.astral.sh/uv/). After [installing uv](https://docs.astral.sh/uv/getting-started/installation/), create the environment with:

```bash
uv sync
```
This creates `.venv` with Python 3.12 and the exact versions recorded in `uv.lock`.

### 2. Download Required Models
To run the code, you need to download (from the repository root):
- An embedding model ```uv run python download_embedding_model.py```
- (Optional) A reranker model ```uv run python download_reranker_model.py```
- A Small Language Model (SLM) ```uv run python download_model.py```

### 3. Create the Vector Index for R.A.G.
Generate the vector index required for the R.A.G. part by running:

```bash
cd bin
uv run python create_vectordb.py
```
You can adjust the parameters using the available command-line options.

### 4. Run the Main Application
Once all models are downloaded and the vector index is created, you can execute the main application (still from `bin/`) with:

```bash
uv run python main.py --model qwen_0.5b
```
`--model` takes one of the keys of `model_dict` in `bin/config.py` (default `phi_int4`).
if you want try the application with a GUI done with Gradio you can execute the following command:

```bash
uv run python main_gradio.py
```
The page opens at http://127.0.0.1:7860. It streams the answer, shows the passages of the guide it was built from (with their page) and how long the answer took, and lets you switch between the models you have downloaded. Add `--no-browser` on a machine without a display.

---

## Development 🛠️

```bash
uv run pytest                 # tests (a few seconds, no model needed)
uv run ruff check . --fix     # lint
uv run ruff format .          # format
```

The thesis experiments ran on the 2024 versions of the libraries (OpenVINO 2024.3, optimum-intel 1.19, transformers 4.44, LangChain 0.2). Those exact versions are recorded in commit `f944926`; the results in `data/` were produced with them.

---

## Future work 🧭

Ideas for the next iterations, not yet started:

- **A newer 0.5B-class model.** Export a more recent small model (qwen3.5 0.5b) with `download_model.py`, add it to `model_dict` in `bin/config.py`, and run the same 25-question evaluation and timing tests on it (`test_SLMs.py`, `time_inference.py`) to compare it with Qwen2.5-0.5B. Use `--output-dir` so the thesis results in `data/` are kept.
- **More and different documents, up to nautical charts.** Today the index is built from a single PDF and only `.pdf` files can be loaded. The first step is several documents of different kinds in one index; the long-term goal is answering from charts and maps, which are not plain text and will need their own way of being read and retrieved.
- **Rethink the model evaluation.** The current flow (answers CSV, then RAGAS metrics in the Colab notebooks under `evaluation/`) dates from the thesis. Review which metrics are used, how they are computed, and whether the evaluation can run locally and reproducibly.
