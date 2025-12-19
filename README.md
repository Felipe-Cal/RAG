# RAG for Payment Risk Investigation

Production-oriented **Retrieval-Augmented Generation (RAG)** and **Multi-Agent** implementations designed for payment systems and fraud detection use cases.

## Overview

This repository demonstrates how to build safe, auditable AI systems for regulated environments where AI assists humans rather than making decisions. The focus is on:

- **Grounded generation** using internal knowledge
- **Safety guardrails** and output validation
- **Explainability** and audit trails
- **Evaluation** and observability hooks

## Notebooks

---

### 1. RAG Payment Safe with Evaluation
`1. rag_payment_safe_with_evaluation.ipynb`

A production-oriented RAG pipeline using **LlamaIndex** with document ingestion, vector retrieval, and evaluation hooks.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG PIPELINE (LlamaIndex)                         │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  OFFLINE PHASE (Batch)                                              │
    │                                                                     │
    │   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
    │   │   Payment    │───▶│    Chunk     │───▶│   Vector Index       │  │
    │   │   Documents  │    │    & Embed   │    │   (ChromaDB)         │  │
    │   └──────────────┘    └──────────────┘    └──────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ONLINE PHASE (Request-time)                                        │
    │                                                                     │
    │   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
    │   │   Analyst    │───▶│   Retrieve   │───▶│   LLM Generation     │  │
    │   │   Question   │    │   Context    │    │   (Grounded)         │  │
    │   └──────────────┘    └──────────────┘    └──────────────────────┘  │
    │                                                      │              │
    │                                                      ▼              │
    │                                           ┌──────────────────────┐  │
    │                                           │   Validation &       │  │
    │                                           │   Audit Logging      │  │
    │                                           └──────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────┘
```

**Key features:**
- Document ingestion and chunking (chunk_size=450, chunk_overlap=50)
- HuggingFace BGE-small-en-v1.5 embeddings (384 dimensions)
- Persistent vector storage with ChromaDB
- Ollama + Llama 3.1 8B for local inference
- Grounded generation with context
- Evaluation and observability hooks

---

### 2. LangChain Payment Safe RAG
`2. langchain_payment_safe_rag.ipynb`

Payment-safe workflow using **LangChain** with multi-step reasoning and safety guardrails.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LANGCHAIN PAYMENT WORKFLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌───────────────────┐
                              │    Transaction    │
                              │      Input        │
                              └─────────┬─────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │     STEP 1: POLICY RETRIEVAL          │
                    │     Retrieve relevant payment and     │
                    │     compliance policies               │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │     STEP 2: RISK CLASSIFICATION       │
                    │     Classify transaction into         │
                    │     risk categories                   │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │     STEP 3: EXPLANATION GENERATION    │
                    │     Generate structured explanation   │
                    │     for flagged transactions          │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │     STEP 4: OUTPUT VALIDATION         │
                    │     • Safety guardrails               │
                    │     • Forbidden action check          │
                    │     • Audit logging                   │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │  Analyst Review   │
                              │  (Human Decision) │
                              └───────────────────┘
```

**Key features:**
- Multi-step reasoning with LangChain chains
- Policy retrieval tools
- Risk classification
- Safety guardrails and output validation

---

### 3. LangChain Router + Validators
`3. langchain_router_validators_rag.ipynb`

Fraud detection workflow with **Router Chain** and **Pydantic Validators** for type-safe processing.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ROUTER CHAIN + VALIDATORS WORKFLOW                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌───────────────────┐
                              │    Transaction    │
                              │      Input        │
                              └─────────┬─────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │         INPUT VALIDATION              │
                    │   Pydantic TransactionInput Model     │
                    │   • Validate amount > 0               │
                    │   • Validate currency (ISO code)      │
                    │   • Validate transaction_type         │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │           ROUTER CHAIN                │
                    │   RunnableBranch - Routes based on    │
                    │   transaction_type                    │
                    └───────────────────┬───────────────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                        │
               ▼                        ▼                        ▼
    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │   WIRE TRANSFER  │    │   CARD PAYMENT   │    │  CRYPTO PAYMENT  │
    │      CHAIN       │    │      CHAIN       │    │      CHAIN       │
    │  Specialized     │    │  Specialized     │    │  Specialized     │
    │  fraud rules     │    │  fraud rules     │    │  fraud rules     │
    └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
               │                        │                        │
               └────────────────────────┼────────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │         OUTPUT VALIDATION             │
                    │   Pydantic FraudAnalysisOutput        │
                    │   • risk_score: 0.0-1.0               │
                    │   • fraud_indicators: list            │
                    │   • recommendation: string            │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │         SAFETY VALIDATION             │
                    │   • Check forbidden action words      │
                    │   • AI recommends, never decides      │
                    └───────────────────────────────────────┘
```

**Key features:**
- Dynamic routing based on transaction type
- Strict Pydantic input/output validation
- Specialized chains per transaction type
- Safety validators for guardrails

---

### 4. LangGraph Multi-Agent System
`4. langgraph_multi_agent_payments.ipynb`

Multi-agent payment processing system using **LangGraph** with specialized agents for compliance, fraud, and risk.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   MULTI-AGENT PAYMENT PROCESSING SYSTEM                     │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌───────────────────┐
                              │  Payment Request  │
                              │     (Input)       │
                              └─────────┬─────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │          SUPERVISOR AGENT             │
                    │   • Routes to specialist agents       │
                    │   • Aggregates findings               │
                    │   • Makes final recommendation        │
                    └───────────────────┬───────────────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                        │
               ▼                        ▼                        ▼
    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │   COMPLIANCE     │    │ FRAUD DETECTION  │    │ RISK ASSESSMENT  │
    │     AGENT        │    │     AGENT        │    │     AGENT        │
    │                  │    │                  │    │                  │
    │ • Sanctions      │    │ • Pattern        │    │ • Risk scoring   │
    │ • AML check      │    │   analysis       │    │ • Policy match   │
    │ • Regulatory     │    │ • Velocity       │    │ • Historical     │
    │   rules          │    │ • Anomaly        │    │   context        │
    └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
               │                        │                        │
               └────────────────────────┼────────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │        AGGREGATED FINDINGS            │
                    │   • Combined risk assessment          │
                    │   • Compliance status                 │
                    │   • Fraud indicators                  │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │        SHARED TOOLS                   │
                    │   • Policy retrieval                  │
                    │   • Customer lookup                   │
                    │   • Sanctions check                   │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │ ANALYST DASHBOARD │
                              │ (Human Decision)  │
                              └───────────────────┘
```

**Key features:**
- Multi-agent orchestration with LangGraph
- Specialized agents (Compliance, Fraud, Risk)
- Agent collaboration and information sharing
- Structured state management

---

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
# For LlamaIndex notebooks (1)
pip install llama-index chromadb llama-index-llms-ollama llama-index-embeddings-huggingface llama-index-vector-stores-chroma

# For LangChain notebooks (2, 3)
pip install langchain langchain-ollama pydantic

# For LangGraph notebooks (4)
pip install langchain langchain-ollama langgraph pydantic
```

## Key Principle

> **AI assists analysts; it does not take payment actions.**

These implementations are designed for regulated environments where explainability, auditability, and human oversight are critical.

## License

MIT
