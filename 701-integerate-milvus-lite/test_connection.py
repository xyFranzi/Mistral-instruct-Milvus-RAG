from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="aari1995/German_Semantic_V3b")

URI = "./milvus_example.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)