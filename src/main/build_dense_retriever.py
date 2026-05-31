import os
from retriv import DenseRetriever

def crtDenseRetriever(retriv_index_name: str, file_path: str, model: str):
    """
    创建密集检索器索引
    :param retriv_index_name: retriv 索引名称（必须与 config 中 retriever_version 一致，如 "dr_corom_emb"）
    :param file_path: 实体 CSV 文件的绝对路径
    :param model: 使用的嵌入模型
    :return: 索引名称 和 模型
    """
    print(f"开始创建密集检索器索引，索引名称: {retriv_index_name}")
    print(f"使用的嵌入模型: {model}")
    print(f"实体数据来源: {file_path}")

    # 创建 DenseRetriever（index_name 是名称，retriv 自动存到 ~/.retriv/collections/{index_name}）
    dr = DenseRetriever(
        index_name=retriv_index_name,          # 必须和 config 里的 retriever_version 一致
        model=model,
        normalize=True,
        max_length=512,
        use_ann=True,
    )

    # 构建索引
    dr.index_file(
        path=file_path,
        embeddings_path=None,
        use_gpu=True,
        batch_size=512,
        show_progress=True,
        callback=lambda doc: {
            "id": doc["id"],
            "text": doc["text"],
        },
    )

    print(f"✅ 索引 '{retriv_index_name}' 创建完成！")
    return retriv_index_name, model


if __name__ == "__main__":
    # === 绝对路径配置（根据你的项目结构调整）===
    ENTITY_CSV_PATH = "/home/a/SSW/RE-MCDF/data/entities.csv"  # 你的实体 CSV
    RETRIEVER_VERSION = "dr_corom_emb"                         # 必须和 config 中一致
    EMBEDDING_MODEL = "/home/a/SSW/RE-MCDF/models/sbert_nlp_corom_sentence-embedding_chinese-base-ecom"

    # 确保 CSV 文件存在
    if not os.path.exists(ENTITY_CSV_PATH):
        raise FileNotFoundError(f"实体文件不存在: {ENTITY_CSV_PATH}")

    # 创建索引
    index_name, model_used = crtDenseRetriever(
        retriv_index_name=RETRIEVER_VERSION,
        file_path=ENTITY_CSV_PATH,
        model=EMBEDDING_MODEL
    )

    # 验证加载
    print("\n🔍 尝试加载索引...")
    try:
        dr = DenseRetriever.load(index_name)
        test_results = dr.search(query="高血压", return_docs=True, cutoff=3)
        print("\n🧪 测试检索结果:")
        for i, res in enumerate(test_results, 1):
            print(f"  {i}. {res['text']} (score: {res['score']:.4f})")
        print("\n✅ 索引验证成功！")
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")