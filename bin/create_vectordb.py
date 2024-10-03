from vectordb import create_vectordb, load_single_document, load_hf_embedding_model, load_ov_embedding_model
from config import pdf_path, vectorstore_path
import argparse
import os
import logging

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Create a vector store from a list of documents")
    parser.add_argument("docs", nargs="*", default=[pdf_path], help="Path(s) to the documents (space-separated)")
    parser.add_argument("--splitter_name", default="RecursiveCharacter", help="Text splitter strategy (e.g., 'RecursiveCharacter', 'Character')")
    parser.add_argument("--chunk_size", type=int, default=400, help="Chunk size in characters")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Number of characters to overlap between chunks")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5", help="Name of the embedding model to use (model_vendor/actual_model_id)")
    parser.add_argument("--vectorstore_path", type=str, default=vectorstore_path, help="Path to save the vector store")
    
    args = parser.parse_args()

    if args.embedding_model.count("/") != 1:
        raise FileNotFoundError("The model name must be in the format model_vendor/actual_model_id")

    # Validate document paths
    for doc in args.docs:
        if not os.path.isfile(doc):
            raise FileNotFoundError(f"Document not found: {doc}")

    # Validate chunking arguments
    if args.chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer")
    if args.chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size")

    logging.info(f"Creating vector store from documents: {args.docs}")

    try:
        create_vectordb(
            args.docs,
            args.splitter_name,
            args.chunk_size,
            args.chunk_overlap,
            args.embedding_model,
            args.vectorstore_path
        )
    except Exception as e:
        logging.error(f"Failed to create vector store: {e}")
        raise



