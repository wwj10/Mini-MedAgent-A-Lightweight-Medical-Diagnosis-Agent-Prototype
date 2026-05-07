import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from episode_runner import EpisodeRunner


case_path = PROJECT_ROOT / "data" / "cases.json"

with case_path.open("r", encoding="utf-8") as f:
    cases = json.load(f)

case = cases[0]

runner = EpisodeRunner(
    case=case,
    max_turns=8,
    use_memory=True,
    memory_mode="all",
    memory_retrieval_mode="embedding",
)

result = runner.run()

output_dir = PROJECT_ROOT / "outputs" / "embedding_memory_test"
output_path = runner.save_transcript(output_dir)

print("\nEmbedding memory 单病例测试完成")
print("结果保存到：", output_path)
