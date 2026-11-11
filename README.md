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
The scripts in `bin/` use relative paths, so run them from inside that folder. Generate the vector index required for the R.A.G. part by running:

```bash
cd bin
uv run python create_vectordb.py
```
You can adjust the parameters using the available command-line options.

### 4. Run the Main Application
Once all models are downloaded and the vector index is created, you can execute the main application (still from `bin/`) with:

```bash
uv run python main.py
```
if you want try the application with a GUI done with Gradio you can execute the following command:

```bash
uv run python main_gradio.py
```
