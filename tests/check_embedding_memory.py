import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from memory_manager import MemoryManager


memory_path = PROJECT_ROOT / "outputs" / "memory" / "medical_memory.json"

if not memory_path.exists():
    raise FileNotFoundError(
        f"没有找到经验库：{memory_path}，请先运行 check_batch_runner.py 或 check_episode_runner.py 生成 memory。"
    )

manager_keyword = MemoryManager(
    memory_path=memory_path,
    retrieval_mode="keyword",
)

manager_embedding = MemoryManager(
    memory_path=memory_path,
    retrieval_mode="embedding",
)

test_queries = [
    {
        "chief_complaint": "发热、咳嗽3天",
        "transcript": [],
    },
    {
        "chief_complaint": "突发胸痛2小时",
        "transcript": [],
    },
    {
        "chief_complaint": "突发呼吸困难伴胸痛1小时",
        "transcript": [],
    },
    {
        "chief_complaint": "多饮、多尿、体重下降1个月",
        "transcript": [],
    },
]

for item in test_queries:
    print("=" * 100)
    print("测试主诉：", item["chief_complaint"])

    print("\n【关键词检索结果】")
    print("-" * 100)
    print(
        manager_keyword.format_relevant_memories(
            chief_complaint=item["chief_complaint"],
            transcript=item["transcript"],
            top_k=3,
            lesson_type_filter="all",
            retrieval_mode="keyword",
        )
    )

    print("\n【Embedding 语义检索结果】")
    print("-" * 100)
    print(
        manager_embedding.format_relevant_memories(
            chief_complaint=item["chief_complaint"],
            transcript=item["transcript"],
            top_k=3,
            lesson_type_filter="all",
            retrieval_mode="embedding",
        )
    )
