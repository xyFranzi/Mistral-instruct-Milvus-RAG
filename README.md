# Mistral-instruct-Milvus-RAG

## Overview
This repository contains tools and scripts for integrating Mistral models with Milvus database for retrieval-augmented generation (RAG). It includes utilities for embedding, querying, evaluating models, and Docker-based deployment setups.

## Key Directories and Files
- **24-model-load/**: Scripts for loading and testing Mistral models.
  - `embedding_ger_sematic_V3b.py`
  - `mistral_7b_test.py`

- **25-prompt-test/**: Scripts for testing prompt-based interactions.
  - `prompt_test.py`, `test_ganz_klein.py`

- **26-mock-tool/**: Mock tools and instruction tests.
  - Includes `test_prompt3.py` and `instruction_test/` subfolder.

- **27-mock-tool/**: Additional mock tools.
  - `de_tool.py`, `p_tool.py`

- **30-integrate-milvus-docker/**: Docker setup for Milvus integration.
  - `Dockerfile`, `docker-compose.yml`
  - `main.py`, `milvus_setup.py`

- **701-integerate-milvus/**: Experimental scripts for Milvus integration.
  - `exp.py`, `milvus_setup.py`

- **701-integerate-milvus-lite/**: Lite version of Milvus integration.
  - `de_tool.py`, `test_connection.py`

- **708-milvus-lite-search/**: Scripts for lightweight Milvus search.
  - `doc_to_milvus.py`, `query_mlivus_lite.py`

- **709-GPT-Evaluator/**: Evaluation scripts for GPT models.
  - `doc_to_milvus.py`, `gpt_eval_4o.py`
  - `test.py`, `tool_usage_check_json.py`

- **711-edit-logic/**: Tools for editing logic in DE tools.
  - `de_tool_ss_web.py`, `de_tool.py`

- **714-tool-json-format/**: Scripts for JSON formatting of tools.
  - `db_ss.py`, `tool_json_test.py`

- **716-de-tool-GPT-eval/**: DE tool evaluation scripts for GPT.
  - `de_tool_eval.py`, `de_tool_to_json.py`

- **almost-version/**: Near-final versions of DE tools and scripts.
  - `de_tool_eval.py`, `doc_to_milvus.py`

- **eval_results/**

- **test_results/**

## Usage
1. Install dependencies:
   ```bash
   uv pip install -r requirements.txt

2. Run scripts

## !Project archived, move to new one - oLLM Test
