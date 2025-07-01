from fastapi import FastAPI
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ 修改这里
from pymilvus import connections

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "LangChain + Milvus 正常运行"}

@app.get("/test")
def test_milvus():
    connections.connect("default", host="milvus", port="19530")

    embedding = HuggingFaceEmbeddings(model_name="aari1995/German_Semantic_V3b")

    vector_db = Milvus(
        embedding_function=embedding,
        collection_name="langchain_demo",
        connection_args={"host": "milvus", "port": "19530"},
    )

    texts = ["Das ist ein Test.", "Künstliche Intelligenz ist faszinierend."]
    metadatas = [{"id": "1"}, {"id": "2"}]

    vector_db.add_texts(texts, metadatas=metadatas)

    result = vector_db.similarity_search("Was ist KI?", k=1)

    return {"result": result[0].page_content}
