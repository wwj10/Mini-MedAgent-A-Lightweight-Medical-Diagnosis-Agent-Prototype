import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from memory_manager import MemoryManager


memory_path = PROJECT_ROOT / "outputs" / "memory" / "medical_memory.json"

if not memory_path.exists():
    raise FileNotFoundError(
        f"没有找到经验库：{memory_path}，请先运行带 memory 的 episode。"
    )

manager = MemoryManager(
    memory_path=memory_path,
    retrieval_mode="embedding",
)

chief_complaint = "发热、咳嗽3天"
transcript = []

print("=" * 80)
print("第 1 次 embedding 检索")
print("=" * 80)

start = time.perf_counter()

memories_1 = manager.retrieve_relevant_memories(
    chief_complaint=chief_complaint,
    transcript=transcript,
    top_k=3,
    lesson_type_filter="all",
    retrieval_mode="embedding",
)

elapsed_1 = time.perf_counter() - start

print(f"检索到 {len(memories_1)} 条经验")
print(f"耗时：{elapsed_1:.4f} 秒")

for memory in memories_1:
    print("-", memory.get("label"), memory.get("content"))

print("\n" + "=" * 80)
print("第 2 次 embedding 检索")
print("=" * 80)

start = time.perf_counter()

memories_2 = manager.retrieve_relevant_memories(
    chief_complaint=chief_complaint,
    transcript=transcript,
    top_k=3,
    lesson_type_filter="all",
    retrieval_mode="embedding",
)

elapsed_2 = time.perf_counter() - start

print(f"检索到 {len(memories_2)} 条经验")
print(f"耗时：{elapsed_2:.4f} 秒")

for memory in memories_2:
    print("-", memory.get("label"), memory.get("content"))

print("\n" + "=" * 80)
print("缓存测试结果")
print("=" * 80)
print(f"第 1 次耗时：{elapsed_1:.4f} 秒")
print(f"第 2 次耗时：{elapsed_2:.4f} 秒")
print("如果第 2 次明显更快，并且出现“使用缓存的 memory embeddings”，说明缓存生效。")
