"""自动生成 .env.example 模板"""
import os

ENV_TEMPLATE = """# ============================================
#   智能导购 Agent - 环境变量配置
# ============================================
# 使用步骤:
#   1. cp .env.example .env
#   2. 填入下方真实值
#   3. 千万不要把 .env 提交到 git！
# ============================================

# ========= 通义千问 API (必填) =========
# 申请地址: https://dashscope.console.aliyun.com/apiKey
DASHSCOPE_API_KEY=sk-your-api-key-here

# ========= Redis (可选) =========
# 无 Redis 时会自动降级为内存模式
REDIS_HOST=localhost
REDIS_PORT=6379

# ========= 数据路径 =========
MODEL_CACHE_DIR=./models
DATA_DIR=./data
DB_PATH=./chroma_db
SQLITE_PATH=./profile.db
FEEDBACK_DB_PATH=./feedback.db
"""

def main():
    target = ".env.example"
    if os.path.exists(target):
        print(f"⚠️ {target} 已存在，跳过")
        return
    with open(target, "w", encoding="utf-8") as f:
        f.write(ENV_TEMPLATE)
    print(f"✅ 已生成 {target}")
    print("📝 下一步: cp .env.example .env 并填入真实 API Key")

if __name__ == "__main__":
    main()
