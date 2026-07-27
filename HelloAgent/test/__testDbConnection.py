"""测试 Neo4j 和 Qdrant 连接性

运行方式（项目根目录）：
    python -m test.__testDbConnection
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

from storage.qdrant_store import QdrantConnectionManager
from storage.neo4j_store import Neo4jGraphStore
from storage.embedding import get_text_embedder, get_dimension


def test_qdrant():
    """测试 Qdrant 连接"""
    print("=" * 50)
    print("🔵 测试 Qdrant 向量数据库连接")
    print("=" * 50)

    try:

        embedder = get_text_embedder()

        # 向量存储（Qdrant - 使用连接管理器避免重复连接）
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        store = QdrantConnectionManager.get_instance(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
            vector_size=get_dimension(getattr(embedder, 'dimension', 384)),
            distance=os.getenv("QDRANT_DISTANCE", "cosine")
        )

        # 健康检查
        healthy = store.health_check()
        print(f"\n  Health Check: {'✅ 通过' if healthy else '❌ 失败'}")

        # 集合详情
        if healthy:
            info = store.get_collection_info()
            print(f"  Points Count: {info.get('points_count', 'N/A')}")
            print(f"  Vectors:      {info.get('vectors_count', 'N/A')}")
            print(f"  Segments:     {info.get('segments_count', 'N/A')}")

        return healthy

    except Exception as e:
        print(f"\n  ❌ 连接失败: {e}")
        return False


def test_neo4j():
    """测试 Neo4j 连接"""
    print("\n" + "=" * 50)
    print("🟢 测试 Neo4j 图数据库连接")
    print("=" * 50)

    try:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        store = Neo4jGraphStore(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database
        )

        # 健康检查
        healthy = store.health_check()
        print(f"\n  Health Check: {'✅ 通过' if healthy else '❌ 失败'}")

        # 统计信息
        if healthy:
            stats = store.get_stats()
            print(f"  Nodes:         {stats.get('total_nodes', 'N/A')}")
            print(f"  Relationships: {stats.get('total_relationships', 'N/A')}")
            print(f"  Entity Nodes:  {stats.get('entity_nodes', 'N/A')}")
            print(f"  Memory Nodes:  {stats.get('memory_nodes', 'N/A')}")

        return healthy

    except Exception as e:
        print(f"\n  ❌ 连接失败: {e}")
        return False


def main():
    print("\n🔌 数据库连接性测试")
    print(f"Python: {sys.version}")

    qdrant_ok = test_qdrant()
    neo4j_ok = test_neo4j()

    print("\n" + "=" * 50)
    print("📊 汇总")
    print("=" * 50)
    print(f"  Qdrant: {'✅ 正常' if qdrant_ok else '❌ 失败'}")
    print(f"  Neo4j:  {'✅ 正常' if neo4j_ok else '❌ 失败'}")

    if qdrant_ok and neo4j_ok:
        print("\n🎉 所有数据库连接正常！")
        return 0
    else:
        print("\n⚠️ 部分数据库连接失败，请检查服务是否已启动：")
        if not qdrant_ok:
            print("   Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        if not neo4j_ok:
            print("   Neo4j:  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5.14")
        return 1


if __name__ == "__main__":
    sys.exit(main())
