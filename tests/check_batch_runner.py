import json
import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from episode_runner import EpisodeRunner


case_path = PROJECT_ROOT / "data" / "cases.json"

with case_path.open("r", encoding="utf-8") as f:
    cases = json.load(f)

output_dir = PROJECT_ROOT / "outputs" / "demo_transcripts"
output_dir.mkdir(parents=True, exist_ok=True)

summary_rows = []


def count_action_labels(action_reviews):
    label_counts = {
        "HIGH_YIELD": 0,
        "REASONABLE": 0,
        "LOW_YIELD": 0,
        "INEFFICIENT": 0,
        "CRITICAL_ERROR": 0,
    }

    for review in action_reviews:
        label = review.get("label", "REASONABLE")
        if label not in label_counts:
            label = "REASONABLE"
        label_counts[label] += 1

    return label_counts


print("=" * 80)
print(f"开始批量运行，共 {len(cases)} 个病例")
print("=" * 80)

for case in cases:
    runner = EpisodeRunner(case=case, max_turns=8)
    result = runner.run()

    transcript_path = runner.save_transcript(output_dir)

    doctor_usage = result["llm_usage"]
    judge_usage = result["judge_llm_usage"]
    process_usage = result["process_grader_usage"]

    label_counts = count_action_labels(result["action_reviews"])

    summary_rows.append({
        "case_id": result["case_id"],
        "chief_complaint": result["chief_complaint"],
        "gold_diagnosis": result["gold_diagnosis"],
        "final_diagnosis": result["final_diagnosis"],

        "score": result["score"],
        "rule_score": result["rule_score"],
        "judge_label": result["judge_label"],
        "judge_reason": result["judge_reason"],

        "turns": result["turns"],
        "total_cost": result["total_cost"],
        "ordered_tests": ";".join(result["ordered_tests"]),

        "process_overall_comment": result["process_overall_comment"],
        "high_yield_count": label_counts["HIGH_YIELD"],
        "reasonable_count": label_counts["REASONABLE"],
        "low_yield_count": label_counts["LOW_YIELD"],
        "inefficient_count": label_counts["INEFFICIENT"],
        "critical_error_count": label_counts["CRITICAL_ERROR"],

        "doctor_llm_call_count": doctor_usage["call_count"],
        "doctor_llm_total_tokens": doctor_usage["total_tokens"],
        "judge_llm_call_count": judge_usage["call_count"],
        "judge_llm_total_tokens": judge_usage["total_tokens"],
        "process_grader_llm_call_count": process_usage["call_count"],
        "process_grader_llm_total_tokens": process_usage["total_tokens"],

        "transcript_path": str(transcript_path),
    })


print("\n" + "=" * 80)
print("批量运行结束")
print("=" * 80)

num_cases = len(summary_rows)

avg_judge_score = sum(row["score"] for row in summary_rows) / num_cases
avg_rule_score = sum(row["rule_score"] for row in summary_rows) / num_cases
avg_turns = sum(row["turns"] for row in summary_rows) / num_cases
avg_cost = sum(row["total_cost"] for row in summary_rows) / num_cases

total_high_yield = sum(row["high_yield_count"] for row in summary_rows)
total_reasonable = sum(row["reasonable_count"] for row in summary_rows)
total_low_yield = sum(row["low_yield_count"] for row in summary_rows)
total_inefficient = sum(row["inefficient_count"] for row in summary_rows)
total_critical_error = sum(row["critical_error_count"] for row in summary_rows)

total_doctor_calls = sum(row["doctor_llm_call_count"] for row in summary_rows)
total_doctor_tokens = sum(row["doctor_llm_total_tokens"] for row in summary_rows)
total_judge_calls = sum(row["judge_llm_call_count"] for row in summary_rows)
total_judge_tokens = sum(row["judge_llm_total_tokens"] for row in summary_rows)
total_process_calls = sum(row["process_grader_llm_call_count"] for row in summary_rows)
total_process_tokens = sum(row["process_grader_llm_total_tokens"] for row in summary_rows)

print(f"病例数量：{num_cases}")
print(f"平均 Judge 得分：{avg_judge_score:.2f}")
print(f"平均规则得分：{avg_rule_score:.2f}")
print(f"平均轮数：{avg_turns:.2f}")
print(f"平均成本：{avg_cost:.2f}")

print("\n动作标签总数：")
print(f"HIGH_YIELD：{total_high_yield}")
print(f"REASONABLE：{total_reasonable}")
print(f"LOW_YIELD：{total_low_yield}")
print(f"INEFFICIENT：{total_inefficient}")
print(f"CRITICAL_ERROR：{total_critical_error}")

print("\nLLM 调用统计：")
print(f"Doctor LLM 总调用次数：{total_doctor_calls}")
print(f"Doctor LLM 总 token：{total_doctor_tokens}")
print(f"Judge LLM 总调用次数：{total_judge_calls}")
print(f"Judge LLM 总 token：{total_judge_tokens}")
print(f"ProcessGrader LLM 总调用次数：{total_process_calls}")
print(f"ProcessGrader LLM 总 token：{total_process_tokens}")

print("\n逐病例结果：")

for row in summary_rows:
    print("-" * 80)
    print("病例ID：", row["case_id"])
    print("主诉：", row["chief_complaint"])
    print("标准诊断：", row["gold_diagnosis"])
    print("Agent诊断：", row["final_diagnosis"])
    print("Judge得分：", row["score"])
    print("规则得分：", row["rule_score"])
    print("Judge标签：", row["judge_label"])
    print("轮数：", row["turns"])
    print("成本：", row["total_cost"])
    print("检查：", row["ordered_tests"])
    print("ProcessGrader总评：", row["process_overall_comment"])
    print("HIGH_YIELD：", row["high_yield_count"])
    print("REASONABLE：", row["reasonable_count"])
    print("LOW_YIELD：", row["low_yield_count"])
    print("INEFFICIENT：", row["inefficient_count"])
    print("CRITICAL_ERROR：", row["critical_error_count"])

summary_path = PROJECT_ROOT / "outputs" / "batch_summary.csv"

with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames = [
        "case_id",
        "chief_complaint",
        "gold_diagnosis",
        "final_diagnosis",

        "score",
        "rule_score",
        "judge_label",
        "judge_reason",

        "turns",
        "total_cost",
        "ordered_tests",

        "process_overall_comment",
        "high_yield_count",
        "reasonable_count",
        "low_yield_count",
        "inefficient_count",
        "critical_error_count",

        "doctor_llm_call_count",
        "doctor_llm_total_tokens",
        "judge_llm_call_count",
        "judge_llm_total_tokens",
        "process_grader_llm_call_count",
        "process_grader_llm_total_tokens",

        "transcript_path",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

print("\n实验汇总表已保存到：")
print(summary_path)
