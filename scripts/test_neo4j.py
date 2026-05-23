"""测试 Neo4j 连接"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

print(f"📂 项目根目录: {ROOT}")
print(f"📄 .env 路径: {ENV_PATH}")
print(f"📄 .env 是否存在: {ENV_PATH.exists()}")

if not ENV_PATH.exists():
    print("❌ .env 文件不存在")
    sys.exit(1)

load_dotenv(dotenv_path=ENV_PATH, override=True)

URI = os.getenv("NEO4J_URI")
# 兼容 NEO4J_USER 和 NEO4J_USERNAME
USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

print(f"\n🔍 读取到的配置:")
print(f"   URI: {URI}")
print(f"   USER: {USER}")
print(f"   PASSWORD: {'***' if PASSWORD else None}")

if not all([URI, USER, PASSWORD]):
    print("\n❌ 配置不完整")
    sys.exit(1)

print(f"\n🔗 尝试连接 Neo4j Aura...")

try:
    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        driver.verify_connectivity()
        print("✅ 连接成功！")
        
        with driver.session() as session:
            msg = session.run("RETURN 'Hello, Neo4j!' AS msg").single()["msg"]
            print(f"📬 响应: {msg}")
            
            count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            print(f"📊 当前节点数: {count}")
            
except Exception as e:
    print(f"❌ 连接失败: {type(e).__name__}: {e}")
    sys.exit(1)
