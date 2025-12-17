# RAG for Payment Risk Investigation

Production-oriented **Retrieval-Augmented Generation (RAG)** implementations designed for payment systems and fraud detection use cases.

## Overview

This repository demonstrates how to build safe, auditable AI systems for regulated environments where AI assists humans rather than making decisions. The focus is on:

- **Grounded generation** using internal knowledge
- **Safety guardrails** and output validation
- **Explainability** and audit trails
- **Evaluation** and observability hooks

## Notebooks

### 1. RAG Payment Safe with Evaluation
`1. rag_payment_safe_with_evaluation.ipynb`

A production-oriented RAG pipeline using **LlamaIndex** with:
- Document ingestion and chunking
- Vector-based retrieval
- Grounded generation with context
- Evaluation and observability hooks

### 2. LangChain Payment Safe RAG
`2. langchain_payment_safe_rag.ipynb`

Payment-safe workflow using **LangChain** with:
- Multi-step reasoning chains
- Policy retrieval tools
- Risk classification
- Safety guardrails and output validation
- Observability and logging for audit

### 3. LangChain Router + Validators
`3. langchain_router_validators_rag.ipynb`

Fraud detection workflow with:
- **Router Chain** - Routes transactions to specialized chains based on type
- **Pydantic Validators** - Strict input/output validation
- **Structured Output** - JSON output parsing with schema validation
- Specialized chains for wire transfers, card payments, and crypto

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally

```bash
# Install Ollama from https://ollama.ai
# Then pull a model:
ollama pull llama3.1:8b
```

## Installation

```bash
# For LlamaIndex notebooks
pip install llama-index chromadb llama-index-llms-ollama llama-index-embeddings-huggingface

# For LangChain notebooks
pip install langchain langchain-ollama
```

## Architecture

All notebooks follow a similar pattern:

1. **Offline Phase** - Load documents, chunk, embed, and build vector index
2. **Online Phase** - Receive query, retrieve context, generate grounded response
3. **Validation Phase** - Validate outputs, ensure safety, log for audit

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Document  │────▶│   Vector    │────▶│   LLM with  │
│   Ingestion │     │   Retrieval │     │   Context   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Validation │
                                        │  & Logging  │
                                        └─────────────┘
```

## Key Principle

> AI assists analysts; it does not take payment actions.

These implementations are designed for regulated environments where explainability, auditability, and human oversight are critical.

## License

MIT
