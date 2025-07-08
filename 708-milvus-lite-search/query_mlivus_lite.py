from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List, Dict, Any
import json
import time
import random
import re
from pymilvus import connections, utility
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json
import os
from pymilvus import MilvusClient

db_path = os.path.expanduser("~/mypython/Mistral-instruct-Milvus-RAG/milvus_german_docs.db")
os.makedirs(os.path.dirname(db_path), exist_ok=True)
client = MilvusClient(db_path)
collection_name = "german_docs"

# Load embedding model
hf_embeddings = HuggingFaceEmbeddings(
    model_name="aari1995/German_Semantic_V3b",
    model_kwargs={"device": "cpu"}  # or "cuda" if you have GPU
)

# Example search
def search_documents(query, top_k=3):
    query_embedding = hf_embeddings.embed_query(query)
    
    results = client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=top_k,
        output_fields=["text", "source"]
    )
    
    return results[0] 

query = "Wer ist Raymond?"
results = search_documents(query, top_k=3)
print(f"\nSuche nach: '{query}'")
print("-" * 50)

for i, result in enumerate(results, 1):
    print(f"Ergebnis {i}:")
    print(f"Text: {result['entity']['text']}")
    print(f"Quelle: {result['entity']['source']}")
    print(f"Ähnlichkeit: {result['distance']:.4f}")
    print("-" * 30)

# Close client
client.close()