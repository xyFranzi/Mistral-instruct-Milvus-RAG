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
import os
from pymilvus import connections, utility, MilvusClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json

MILVUS_HOST = "milvus"
MILVUS_PORT = "19530"
DEFAULT_COLLECTION = "german_docs"
EMBEDDING_MODEL = "aari1995/German_Semantic_V3b"

class MilvusSimilaritySearchTool(BaseTool):
    """
    Milvus Lite 语义相似性搜索工具
    使用 MilvusClient 直接连接到本地 .db 文件
    """
    name: str = "similarity_search"
    description: str = (
        "Führe semantische Ähnlichkeitssuche in der Milvus Lite-Kollektion durch. "
        "Eingabe: Query-String; Ausgabe: JSON mit Dokumentfragmenten und Scores."
    )

    _client: Any = None
    _embedder: Any = None
    collection_name: str = "german_docs"
    k: int = 3

    def __init__(self,
                 collection_name: str = "german_docs",
                 db_path: str = None,
                 embedding_model: str = "aari1995/German_Semantic_V3b",
                 embedding_device: str = "cpu",
                 k: int = 3):
        super().__init__()
        
        # Set default db_path to the actual location of the database
        if db_path is None:
            # Get the absolute path to the database file in the workspace root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_root = os.path.dirname(current_dir)  # Go up one level from 711-edit-logic
            db_path = os.path.join(workspace_root, "milvus_german_docs.db")
        
        # Ensure the path is absolute
        db_path = os.path.abspath(db_path)
        
        # Check if database file exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Milvus database file not found: {db_path}")
        
        print(f"Using Milvus database: {db_path}")
        
        # Initialize MilvusClient directly
        self._client = MilvusClient(db_path)
        self.collection_name = collection_name
        
        # Check if collection exists
        collections = self._client.list_collections()
        if collection_name not in collections:
            raise ValueError(f"Milvus Lite collection '{collection_name}' not found in {db_path}. Available: {collections}")

        # Load Embeddings
        self._embedder = HuggingFaceEmbeddings(
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
        使用 MilvusClient 进行相似性检索，格式化返回 JSON 字符串
        """
        # Embed the query
        query_embedding = self._embedder.embed_query(query)
        
        # Execute search using MilvusClient
        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=self.k,
            output_fields=["source", "text"]  # Specify which fields to return
        )

        docs_out: List[Dict[str, Any]] = []
        
        # Process results
        if results and len(results) > 0:
            for hit in results[0]:  # results[0] contains the hits for our single query
                docs_out.append({
                    "score": float(hit.get("distance", 0.0)),  # Milvus returns distance, lower is better
                    "source": hit.get("entity", {}).get("source", "Unknown"),
                    "content": hit.get("entity", {}).get("text", "")
                })

        output = {
            "tool": self.name,
            "query": query,
            "found_documents": len(docs_out),
            "documents": docs_out,
            "search_type": "semantic_similarity"
        }
        return json.dumps(output, ensure_ascii=False, indent=2)