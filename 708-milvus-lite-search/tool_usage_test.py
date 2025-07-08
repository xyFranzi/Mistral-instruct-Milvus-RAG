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

# db_path = os.path.expanduser("~/mypython/Mistral-instruct-Milvus-RAG/milvus_german_docs.db")
# os.makedirs(os.path.dirname(db_path), exist_ok=True)
# client = MilvusClient(db_path)
# collection_name = "german_docs"

# # Load embedding model
# hf_embeddings = HuggingFaceEmbeddings(
#     model_name="aari1995/German_Semantic_V3b",
#     model_kwargs={"device": "cpu"}  # or "cuda" if you have GPU
# )

# # Example search
# def search_documents(query, top_k=3):
#     query_embedding = hf_embeddings.embed_query(query)
    
#     results = client.search(
#         collection_name=collection_name,
#         data=[query_embedding],
#         limit=top_k,
#         output_fields=["text", "source"]
#     )
    
#     return results[0] 

# query = "Wer ist Raymond?"
# results = search_documents(query, top_k=3)
# print(f"\nSuche nach: '{query}'")
# print("-" * 50)

# for i, result in enumerate(results, 1):
#     print(f"Ergebnis {i}:")
#     print(f"Text: {result['entity']['text']}")
#     print(f"Quelle: {result['entity']['source']}")
#     print(f"Ähnlichkeit: {result['distance']:.4f}")
#     print("-" * 30)

# # Close client
# client.close()

class MilvusSimilaritySearchTool(BaseTool):
    """
    Lokales Milvus Lite semantisches Ähnlichkeitssuch-Tool
    Verwendet MilvusClient auf einer lokalen .db-Datei.
    Eingabe: Query-String; Ausgabe: JSON mit Dokumentfragmenten und Scores.
    """
    name: str = "similarity_search"
    description: str = (
        "Führe semantische Ähnlichkeitssuche in der lokalen Milvus-Lite-Datenbank durch. "
        "Eingabe: Query-String; Ausgabe: JSON mit Dokumentfragmenten und Scores."
    )

    def __init__(
        self,
        db_path: str = os.path.expanduser("~/mypython/Mistral-instruct-Milvus-RAG/milvus_german_docs.db"),
        collection_name: str = "german_docs",
        embedding_model: str = "aari1995/German_Semantic_V3b",
        embedding_device: str = "cpu",
        k: int = 3
    ):
        super().__init__()
        # Pfad zur DB anlegen und Client initialisieren
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        # Embedding-Modell laden
        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": embedding_device}
        )
        self.k = k

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Führt semantische Suche mit MilvusClient auf local .db durch und formatiert JSON-Ausgabe.
        """
        # Query einbetten
        query_embedding = self.embedding.embed_query(query)
        # Suche ausführen
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=self.k,
            output_fields=["text", "source"]
        )
        # Ergebnisse formatieren
        docs_out: List[Dict[str, Any]] = []
        for hit in results[0]:
            entity = hit['entity']
            docs_out.append({
                "score": float(hit['distance']),
                "source": entity.get('source'),
                "content": entity.get('text')
            })

        output = {
            "tool": self.name,
            "query": query,
            "found_documents": len(docs_out),
            "documents": docs_out,
            "search_type": "semantic_similarity"
        }
        return json.dumps(output, ensure_ascii=False, indent=2)