from pymilvus import connections
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1) 统一连接配置
MILVUS_HOST = "milvus"  # 如果在 Docker Compose 中，使用服务名
MILVUS_PORT = "19530"

# 先测试连接
try:
    connections.connect(
        alias="default", 
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        timeout=10  # 增加超时时间
    )
    print("成功连接到 Milvus")
except Exception as e:
    print(f"连接 Milvus 失败: {e}")
    print("尝试使用 localhost...")
    MILVUS_HOST = "localhost"
    try:
        connections.connect(
            alias="default", 
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            timeout=10
        )
        print("成功连接到 localhost Milvus")
    except Exception as e2:
        print(f"localhost 连接也失败: {e2}")
        exit(1)

# 2) 加载 Embedding 模型
print("加载 Embedding 模型...")
embedding = HuggingFaceEmbeddings(
    model_name="aari1995/German_Semantic_V3b",
    model_kwargs={"device": "cpu"}
)

# 3) 绑定已有的 Collection（使用统一的连接配置）
print("连接向量数据库...")
vector_db = Milvus(
    embedding_function=embedding,
    collection_name="ulysses_docs",   # 已存在的集合
    connection_args={
        "host": MILVUS_HOST,  # 使用统一的主机地址
        "port": MILVUS_PORT
    }
)

# 4) 相似度检索
print("执行查询...")
query = "in wie viel Teile ist der Roman gegliedert"

results = vector_db.similarity_search_with_score(query, k=3)
    
# 5) 打印结果
print(f"\n查询: {query}")
print(f"找到 {len(results)} 个结果:\n")
    
for i, (doc, score) in enumerate(results, 1):
    print(f"结果 {i}:")
    print(f"  相似度分数: {score:.3f}")
    print(f"  来源: {doc.metadata.get('source', 'N/A')}")
    print(f"  内容: {doc.page_content[:200]}...")
    print("-" * 50)
