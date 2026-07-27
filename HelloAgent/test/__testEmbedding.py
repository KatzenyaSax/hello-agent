"""
测试 Embedding：看看文字是如何被转成向量的

用法：
    cd HelloAgent
    python -m test.__testEmbedding

说明：
    - 优先使用 .env 配置的 dashscope（需网络）
    - 如果网络不通，自动回退到本地 TF-IDF 演示
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def print_vector_info(label, vec, n=8):
    v = np.array(vec)
    print(f"  [{label}]")
    print(f"    维度: {len(v)}")
    print(f"    前{n}个值: [{', '.join(f'{x:.4f}' for x in v[:n])} ...]")
    print(f"    统计: min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}  std={v.std():.4f}")
    print(f"    非零比例: {(v != 0).sum() / len(v):.1%}")
    return v


def demo_with_tfidf():
    """使用 TF-IDF 演示（纯本地，无需网络）"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_texts = [
        "用户张三是一名Python开发者，专注于机器学习和数据分析",
        "张三擅长Python编程，主要做数据科学方向",
        "李四是前端工程师，擅长React和Vue.js开发",
        "今天天气很好，适合出去散步",
        "The user prefers concise answers",
        "用户偏好简洁的回答风格",
        "python developer machine learning data analysis",
    ]

    # TF-IDF：用 char_wb analyzer 处理中文（单字 + 二元组）
    vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(1, 2),
        lowercase=True
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    print("=" * 60)
    print("Embedding 工作原理演示 (TF-IDF 本地模式)")
    print("=" * 60)
    print(f"\n模型: TF-IDF (sklearn)")
    print(f"词汇表大小: {len(feature_names)}")

    # ---- 第 1 步：看一条文本被转成了什么 ----
    print(f"\n{'─' * 60}")
    print("第1步：文本 -> 向量")
    print(f"{'─' * 60}")

    idx = 0
    vec = tfidf_matrix[idx].toarray()[0]
    print(f'\n  原文: "{all_texts[idx]}"')
    print(f"  向量维度: {len(vec)} (每个维度 = 词汇表中的一个词)")

    # 找个有值的维度展示
    nonzero_indices = np.where(vec > 0)[0][:2]
    for idx_nz in nonzero_indices:
        print(f'    "{feature_names[idx_nz]}" -> {vec[idx_nz]:.4f}')
    print(f'    其余 {len(vec) - len(nonzero_indices)} 个维度 = 0 (未出现在该文本中)')

    # ---- 第 2 步：相似度对比 ----
    print(f"\n{'─' * 60}")
    print("第2步：余弦相似度 - 语义相近的文字向量也相近")
    print(f"{'─' * 60}")

    base_vec = tfidf_matrix[0].toarray()[0]  # "用户张三是一名Python开发者..."
    print(f'\n  基准: "{all_texts[0]}"')
    print(f'\n  {"相似度":>8s}  文本')
    print(f'  {"─" * 8}   {"─" * 50}')
    for i, t in enumerate(all_texts):
        sim = cosine_similarity(base_vec, tfidf_matrix[i].toarray()[0])
        bar = "#" * int(sim * 40)
        print(f"  {sim:>7.4f}   {t[:50]:50s}  {bar}")

    # ---- 第 3 步：跨语言语义 ----
    print(f"\n{'─' * 60}")
    print("第3步：跨语言/同义表达")
    print(f"{'─' * 60}")

    # 用 TF-IDF 对两条新查询做 transform（注意：TF-IDF 对未见词直接忽略）
    pairs = [
        ("中文同义", "我偏好简短的回复"),
        ("英文翻译", "The user prefers concise answers"),
        ("完全不相关", "今天天气真不错"),
    ]
    query = "用户喜欢简洁的回答"
    query_vec = vectorizer.transform([query]).toarray()[0]

    print(f'\n  查询: "{query}"')
    print(f'\n  {"相似度":>8s}  文本')
    print(f'  {"─" * 8}   {"─" * 50}')
    for label, t in pairs:
        tv = vectorizer.transform([t]).toarray()[0]
        sim = cosine_similarity(query_vec, tv)
        bar = "#" * max(0, int(sim * 40))
        print(f"  {sim:>7.4f}  [{label}] {t:45s}  {bar}")

    # ---- 第 4 步：向量到底是什么 ----
    print(f"\n{'─' * 60}")
    print("第4步：向量到底是什么？")
    print(f"{'─' * 60}")
    print(f"""
    TF-IDF 的每个维度对应词汇表中的一个词，值是"该词在这条文本中的重要性"。

    比如 "{all_texts[0][:20]}..." 的向量:
    - 有值的维度只有 {(vec > 0).sum()} 个（文本中出现的字/二元组）
    - 其余 {len(vec) - (vec > 0).sum()} 个维度 = 0（未出现）

    而真正的深度学习 Embedding（如 dashscope / MiniLM）:
    - 维度数固定（384 或 1024），不随词汇表大小变化
    - 每个维度没有"对应某个词"的含义，是整个语义的压缩表示
    - 但效果类似：相似文本 → 向量夹角小，不相似 → 向量夹角大
""")

    print("=" * 60)
    print("演示完成")
    print("=" * 60)


def demo_with_real_embedder():
    """使用真实的嵌入模型（dashscope 或 local transformer）"""
    from storage.embedding import get_text_embedder, get_dimension

    embedder = get_text_embedder()
    dim = get_dimension()

    print("=" * 60)
    print("Embedding 工作原理演示 (真实模型)")
    print("=" * 60)
    print(f"\n模型: {type(embedder).__name__}")
    print(f"维度: {dim}")

    texts = [
        "用户张三是一名Python开发者，专注于机器学习和数据分析",
        "张三擅长Python编程，主要做数据科学方向",
        "李四是前端工程师，擅长React和Vue.js开发",
        "今天天气很好，适合出去散步",
        "The user prefers concise answers",
    ]

    vectors = [np.array(embedder.encode(t)) for t in texts]

    # 单条向量信息
    print(f"\n{'─' * 60}")
    print(f'原文: "{texts[0]}"')
    print(f"{'─' * 60}")
    print_vector_info("向量", vectors[0])

    # 相似度矩阵
    print(f"\n{'─' * 60}")
    print(f"相似度矩阵 (以第1条为基准)")
    print(f"{'─' * 60}")
    base = vectors[0]
    for i, t in enumerate(texts):
        sim = cosine_similarity(base, vectors[i])
        bar = "#" * int(sim * 40)
        print(f"  {sim:.4f}  {t[:55]:55s}  {bar}")

    print("=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    try:
        demo_with_real_embedder()
    except Exception as e:
        print(f"[真实模型不可用: {e}]")
        print("回退到 TF-IDF 本地演示...\n")
        demo_with_tfidf()
