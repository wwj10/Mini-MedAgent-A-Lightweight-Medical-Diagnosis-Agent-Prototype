import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from process_grader import ProcessGrader


transcript_path = PROJECT_ROOT / "outputs" / "demo_transcripts" / "case_001_transcript.json"

if not transcript_path.exists():
    raise FileNotFoundError(
        f"没有找到 {transcript_path}，请先运行 check_batch_runner.py 或 check_episode_runner.py。"
    )

with transcript_path.open("r", encoding="utf-8") as f:
    case_result = json.load(f)

grader = ProcessGrader()

grading = grader.grade_actions(
    chief_complaint=case_result["chief_complaint"],
    gold_diagnosis=case_result["gold_diagnosis"],
    final_diagnosis=case_result["final_diagnosis"],
    judge_score=case_result["score"],
    total_cost=case_result["total_cost"],
    transcript=case_result["transcript"],
)

print("=" * 80)
print("ProcessGrader 总评：")
print(grading["overall_comment"])
print("=" * 80)

for review in grading["action_reviews"]:
    print(f"第 {review['turn']} 轮")
    print("动作类型：", review["action_type"])
    print("内容：", review["content"])
    print("标签：", review["label"])
    print("理由：", review["reason"])
    print("建议：", review["suggestion"])
    print("-" * 80)

print("ProcessGrader LLM usage:")
print(grader.get_usage())
