# build_sparse_retriever.py
from retriv import SparseRetriever

def build_sparse_retriever():
    entities = []
    
    with open("/home/a/SSW/RE-MCDF/data/KG_entities2id_merge.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                entity_name = parts[0]
                entities.append({
                    "id": str(i), 
                    "text": entity_name
                })
    
    print(f"总共找到 {len(entities)} 个实体")
    
    # 构建稀疏检索器（BM25）
    retriever = SparseRetriever(
        index_name="bm25_merge_kg"
    )
    
    print("开始构建稀疏检索器索引...")
    retriever.index(entities)
    print("稀疏检索器索引构建完成！")
    
    # 测试
    test_results = retriever.search("高血压", return_docs=True, cutoff=3)
    print("测试检索结果:", test_results)

if __name__ == "__main__":
    build_sparse_retriever()