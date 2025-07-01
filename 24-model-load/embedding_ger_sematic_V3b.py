from langchain.vectorstores import Milvus
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os

PDF_DIR = "./pdf"
COLLECTION_NAME = "german_pdf_collection"

embedding_model = HuggingFaceEmbeddings(model_name="aari1995/German_Semantic_V3b")

# load and chunk
all_docs = []
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(PDF_DIR, filename))
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)

print(f"loading all {len(all_docs)} chunks, start embedding and save to Milvus...")

vectorstore = Milvus.from_documents(
    documents=all_docs,
    embedding=embedding_model,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name=COLLECTION_NAME
)

print(f"embedding done, saved to Milvus collection: {COLLECTION_NAME}")
