# Navigating the Seas of AI: Deploying Small Language Models on Compact Edge Computers for Maritime Applications 🌊⛵

This repository contains the code for creating a chatbot based on a Small Language Model (SLM) integrated with Retrieval-Augmented Generation (R.A.G.), optimized to run on an Intel CPU using the [OpenVINO toolkit](https://github.com/openvinotoolkit/openvino).

Check [here](https://docs.openvino.ai/2024/about-openvino/release-notes-openvino/system-requirements.html) to see if your device supports OpenVINO.

This project was developed as my final work for the MSc in Computer Science, Artificial Intelligence track, at Università di Genova.

---

## Getting Started

### 1. Install Dependencies
First, download all the required dependencies by running:

```bash
pip install -r requirements.txt
```
### 2. Download Required Models
To run the code, you need to download:
- An embedding model
- (Optional) A reranker model
- A Small Language Model (SLM)

### 3. Create the Vector Index for R.A.G.
Generate the vector index required for the R.A.G. part by running:

```bash
python ./bin/create_vectordb.py
```
You can adjust the parameters using the available command-line options.

### 4. Run the Main Application
Once all models are downloaded and the vector index is created, you can execute the main application with:

```python ./bin/main.py```
