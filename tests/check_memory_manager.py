import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from memory_manager import MemoryManager


case_result_path = PROJECT_ROOT / "outputs" / "demo_transcripts" / "case_001_transcript.json"
memory_path = PROJECT_ROOT / "outputs" / "memory" / "medical_memory.json"

if not case_result_path.exists():
    raise FileNotFoundError(
        f"没有找到 {case_result_path}，请先运行 check_episode_runner.py。"
    )

with case_result_path.open("r", encoding="utf-8") as f:
    case_result = json.load(f)

manager = MemoryManager(memory_path=memory_path)

before_count = manager.count()
added_count = manager.add_from_case_result(case_result)
manager.save()
after_count = manager.count()

print("=" * 80)
print("MemoryManager 测试完成")
print("=" * 80)
print("原有经验数量：", before_count)
print("本次新增经验数量：", added_count)
print("当前经验总数：", after_count)
print("经验文件路径：", memory_path)

print("\n检索与“发热、咳嗽3天”相关的经验：")
print("-" * 80)

memory_text = manager.format_relevant_memories(
    chief_complaint="发热、咳嗽3天",
    transcript=[],
    top_k=5,
)

print(memory_text)
