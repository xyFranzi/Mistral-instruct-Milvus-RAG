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
from pydantic import PrivateAttr

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

    collection_name: str
    embedding_model: str
    embedding_device: str
    k: int

    _client: MilvusClient = PrivateAttr()
    _embedder: HuggingFaceEmbeddings = PrivateAttr()


    def __init__(
        self,
        db_path: str = os.path.expanduser("~/mypython/Mistral-instruct-Milvus-RAG/milvus_german_docs.db"),
        collection_name: str = "german_docs",
        embedding_model: str = "aari1995/German_Semantic_V3b",
        embedding_device: str = "cpu",
        k: int = 3
    ):
        super().__init__(
            collection_name=collection_name,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            k=k
        )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._client = MilvusClient(db_path)
        # self.collection_name = collection_name

        # Embedding-Modell laden
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
        Führt semantische Suche mit MilvusClient auf local .db durch und formatiert JSON-Ausgabe.
        """
        # Query einbetten
        query_embedding = self._embedder.embed_query(query)
        # Suche ausführen
        results = self._client.search(
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
    

class RAGToolIntegrationTester:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0.1,
            num_predict= 512 # limit output length
        )
        self.similarity_search_tool = MilvusSimilaritySearchTool()

        self.system_prompt = """
        Du bist ein KI-Assistent, der Fragen zu deutschen Texten beantworten kann.
        Du hast Zugriff auf eine lokale Milvus-Lite-Datenbank, die deutsche Dokumente enthält.
        Deine Aufgabe: Nutze die bereitgestellten Dokumente und beantworte die Frage präzise.
        """

        self.test_query = [
            "Wer ist Meursault und welche Rolle spielt er in 'Der Fremde'?",
            "Wie zeigt Camus den Existentialismus und das Gefühl der Absurdität in 'Der Fremde'?",
            "Was ist der Haputstadt von Deutschland?",
            "Was ist der Unterschied zwischen einem Hund und einer Katze?",
        ]
    
    def test_workflow(self):
        for query in self.test_query:
            print(f"\n{'-'*20}\nTestanfrage: {query}")
            # Semantische Suche
            tool_output = self.similarity_search_tool._run(query)
            print("Ähnlichkeitssuche-Ergebnis:")
            print(tool_output)
            # LLM-Antwort generieren
            prompt = (
                self.system_prompt + "\nDokumente:\n" + tool_output +
                f"\nBeantworte die Frage: {query}"
            )
            # Convert prompt to a list of HumanMessage objects
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Dokumente:\n{tool_output}\nBeantworte die Frage: {query}")
            ]
            response = self.llm.invoke(messages)
            print("LLM-Antwort:")
            print(response)

    def run_comprehensive_test(self):
        print(f"Testing model: {self.model_name}")
        print("=" * 50)

        overall_start = time.time()

        self.test_workflow()

        total_time = time.time() - overall_start

        print(f"\nTotal time for all tests: {total_time:.2f} seconds")
        print("=" * 50)


if __name__ == "__main__":
    # Multiple models can be tested here
    models_to_test = [
        "mistral:7b-instruct-v0.3-q5_0",
        "mistral-small:24b",
        # "xxx",
    ]
    
    for model in models_to_test:
        print(f"\n{'='*20} Testing model: {model} {'='*20}")
        tester = RAGToolIntegrationTester(model)
        tester.run_comprehensive_test()
        
        if len(models_to_test) > 1:
            print(f"\n...")
            time.sleep(5)