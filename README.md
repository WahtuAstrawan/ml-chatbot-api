# Ramayana Kakawin RAG API

A Retrieval-Augmented Generation (RAG) API built with FastAPI that allows users to ask questions about the **Ramayana Kakawin** and receive answers grounded in the original text.

Instead of letting the LLM guess, the system retrieves relevant passages from the Kakawin dataset and provides them as context to the model before generating an answer.

For example:

User question:

> Who married Sinta?

The system retrieves the relevant **kakawin chapter passages** describing the marriage of Rama and Sinta, then the LLM generates an answer using those references.

---

## Architecture Overview

The system combines semantic search and LLM reasoning.

1. User sends a question
2. Question is translated (if necessary)
3. Question embedding is generated
4. FAISS searches the vector database for relevant Kakawin passages
5. Retrieved passages are provided as context to the LLM
6. Gemini generates the final answer based on those references

This prevents hallucination and keeps answers grounded in the source text.

---

## Tech Stack

* **Python**
* **FastAPI** – API framework
* **FAISS** – vector similarity search
* **Cohere Embeddings** – document embeddings
* **Sentence Transformers** – additional embedding utilities
* **Gemini API** – LLM for answer generation
* **Deep Translator** – multilingual question translation

---
