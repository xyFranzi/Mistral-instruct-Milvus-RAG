from pymilvus import connections, Collection, utility
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

MILVUS_HOST = "milvus" 
MILVUS_PORT = "19530"

# 1) 连接 Milvus
print("Connecting Milvus...")
connections.connect(
    alias="default", 
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

# 2) 检查现有集合
collections = utility.list_collections()
print(f"All collections: {collections}")

# 找到正确的集合名
target_collection = None
possible_names = ["der_fremde_docs"]
for name in possible_names:
    if name in collections:
        target_collection = name
        print(f"Collection found: {target_collection}")
        break

if not target_collection:
    print("Colllection not found. Available collections:")
    for col in collections:
        print(f"  - {col}")
    exit(1)

# 3) 检查集合内容
collection = Collection(target_collection)
print(f"Collection: '{target_collection}' Records: {collection.num_entities}")

if collection.num_entities == 0:
    print("Empty collection.")
    exit(1)

# 4) 加载相同的 Embedding 模型
print("Loading HuggingFace Embeddings...")
hf_embeddings = HuggingFaceEmbeddings(
    model_name="aari1995/German_Semantic_V3b",
    model_kwargs={"device": "cpu"}
)

# 5) 连接向量数据库
print("Initializing Milvus vector store...")
vs = Milvus(
    embedding_function=hf_embeddings,
    collection_name=target_collection,  # 使用找到的集合名
    connection_args={
        "host": MILVUS_HOST,
        "port": MILVUS_PORT
    }
)

# 6) 测试多个查询
queries = [
    "In wie viel Teile ist der Roman gegliedert?",
    "Ist Meursault mit seinem Leben zufrieden?",
    "wer ist Raymond"
]

print("\n" + "="*60)
print("Starting queries...")

for i, query in enumerate(queries, 1):
    print(f"\nQ: {i}: {query}")
    print("-" * 40)
    
    try:
        # # 先试试不带分数的搜索
        # results_simple = vs.similarity_search(query, k=3)
        # print(f"简单搜索结果数: {len(results_simple)}")
        
        # # 再试试带分数的搜索
        results_with_score = vs.similarity_search_with_score(query, k=3)
        print(f"Result number: {len(results_with_score)}")
        
        if results_with_score:
            for j, (doc, score) in enumerate(results_with_score, 1):
                print(f"  Outcome: {j}:")
                print(f"    Score: {score:.4f}")
                print(f"    Quelle: {doc.metadata.get('source', 'N/A')}")
                print(f"    Inhalt: {doc.page_content[:150]}...")
                print()
        else:
            print("  No results found for this query.")
            
            print("  Trying broader search...")
            broad_results = vs.similarity_search(query, k=10)
            if broad_results:
                print(f"  {len(broad_results)} results found in broader search:")
                for j, doc in enumerate(broad_results[:3], 1):
                    print(f"    Result {j}: {doc.page_content[:100]}...")
    
    except Exception as e:
        print(f"  query failed {e}")

print("\nDone!")