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

# 先跑第一个病例：社区获得性肺炎
case = cases[0]

runner = EpisodeRunner(case=case, max_turns=8)
result = runner.run()

output_dir = PROJECT_ROOT / "outputs" / "demo_transcripts"
output_path = runner.save_transcript(output_dir)

print("\nTranscript 已保存到：")
print(output_path)
