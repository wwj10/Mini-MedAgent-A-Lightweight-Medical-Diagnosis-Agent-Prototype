import json
import sys
import csv
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from episode_runner import EpisodeRunner


case_path = PROJECT_ROOT / "data" / "cases.json"
memory_path = PROJECT_ROOT / "outputs" / "memory" / "medical_memory.json"
memory_backup_path = PROJECT_ROOT / "outputs" / "memory" / "medical_memory_backup_before_compare.json"

with case_path.open("r", encoding="utf-8") as f:
    cases = json.load(f)


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


def run_condition(condition_name: str, use_memory: bool):
    print("\n" + "=" * 100)
    print(f"开始实验条件：{condition_name}")
    print(f"use_memory = {use_memory}")
    print("=" * 100)

    output_dir = PROJECT_ROOT / "outputs" / condition_name / "demo_transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for case in cases:
        runner = EpisodeRunner(
            case=case,
            max_turns=8,
            use_memory=use_memory,
        )

        result = runner.run()
        transcript_path = runner.save_transcript(output_dir)

        label_counts = count_action_labels(result["action_reviews"])

        doctor_usage = result["llm_usage"]
        judge_usage = result["judge_llm_usage"]
        process_usage = result["process_grader_usage"]

        rows.append({
            "condition": condition_name,
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

            "high_yield_count": label_counts["HIGH_YIELD"],
            "reasonable_count": label_counts["REASONABLE"],
            "low_yield_count": label_counts["LOW_YIELD"],
            "inefficient_count": label_counts["INEFFICIENT"],
            "critical_error_count": label_counts["CRITICAL_ERROR"],
            "process_overall_comment": result["process_overall_comment"],

            "doctor_llm_call_count": doctor_usage["call_count"],
            "doctor_llm_total_tokens": doctor_usage["total_tokens"],
            "judge_llm_call_count": judge_usage["call_count"],
            "judge_llm_total_tokens": judge_usage["total_tokens"],
            "process_grader_llm_call_count": process_usage["call_count"],
            "process_grader_llm_total_tokens": process_usage["total_tokens"],

            "use_memory": result["use_memory"],
            "memory_added_count": result["memory_added_count"],
            "memory_total_count": result["memory_total_count"],

            "transcript_path": str(transcript_path),
        })

    return rows


def summarize(rows, title):
    num_cases = len(rows)

    avg_score = sum(row["score"] for row in rows) / num_cases
    avg_turns = sum(row["turns"] for row in rows) / num_cases
    avg_cost = sum(row["total_cost"] for row in rows) / num_cases

    total_high_yield = sum(row["high_yield_count"] for row in rows)
    total_reasonable = sum(row["reasonable_count"] for row in rows)
    total_low_yield = sum(row["low_yield_count"] for row in rows)
    total_inefficient = sum(row["inefficient_count"] for row in rows)
    total_critical_error = sum(row["critical_error_count"] for row in rows)

    total_doctor_tokens = sum(row["doctor_llm_total_tokens"] for row in rows)
    total_judge_tokens = sum(row["judge_llm_total_tokens"] for row in rows)
    total_process_tokens = sum(row["process_grader_llm_total_tokens"] for row in rows)

    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)
    print(f"病例数量：{num_cases}")
    print(f"平均 Judge 得分：{avg_score:.2f}")
    print(f"平均轮数：{avg_turns:.2f}")
    print(f"平均检查成本：{avg_cost:.2f}")
    print(f"HIGH_YIELD 总数：{total_high_yield}")
    print(f"REASONABLE 总数：{total_reasonable}")
    print(f"LOW_YIELD 总数：{total_low_yield}")
    print(f"INEFFICIENT 总数：{total_inefficient}")
    print(f"CRITICAL_ERROR 总数：{total_critical_error}")
    print(f"Doctor tokens：{total_doctor_tokens}")
    print(f"Judge tokens：{total_judge_tokens}")
    print(f"ProcessGrader tokens：{total_process_tokens}")

    return {
        "condition": title,
        "num_cases": num_cases,
        "avg_score": avg_score,
        "avg_turns": avg_turns,
        "avg_cost": avg_cost,
        "total_high_yield": total_high_yield,
        "total_reasonable": total_reasonable,
        "total_low_yield": total_low_yield,
        "total_inefficient": total_inefficient,
        "total_critical_error": total_critical_error,
        "total_doctor_tokens": total_doctor_tokens,
        "total_judge_tokens": total_judge_tokens,
        "total_process_tokens": total_process_tokens,
    }


# 备份当前 memory，避免覆盖你之前积累的结果
if memory_path.exists():
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(memory_path, memory_backup_path)
    print("已备份原 memory 到：", memory_backup_path)

# 条件 A：不启用 memory
no_memory_rows = run_condition(
    condition_name="no_memory",
    use_memory=False,
)

# 条件 B：启用 memory。为了公平，从空 memory 开始
if memory_path.exists():
    memory_path.unlink()

memory_rows = run_condition(
    condition_name="with_memory",
    use_memory=True,
)

# 汇总
no_memory_summary = summarize(no_memory_rows, "No Memory")
memory_summary = summarize(memory_rows, "With Memory")

# 保存逐病例对比结果
all_rows = no_memory_rows + memory_rows
compare_path = PROJECT_ROOT / "outputs" / "compare_memory_summary.csv"

fieldnames = [
    "condition",
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

    "high_yield_count",
    "reasonable_count",
    "low_yield_count",
    "inefficient_count",
    "critical_error_count",
    "process_overall_comment",

    "doctor_llm_call_count",
    "doctor_llm_total_tokens",
    "judge_llm_call_count",
    "judge_llm_total_tokens",
    "process_grader_llm_call_count",
    "process_grader_llm_total_tokens",

    "use_memory",
    "memory_added_count",
    "memory_total_count",

    "transcript_path",
]

with compare_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

# 保存条件级汇总
summary_path = PROJECT_ROOT / "outputs" / "compare_memory_condition_summary.csv"

with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
    fieldnames_summary = [
        "condition",
        "num_cases",
        "avg_score",
        "avg_turns",
        "avg_cost",
        "total_high_yield",
        "total_reasonable",
        "total_low_yield",
        "total_inefficient",
        "total_critical_error",
        "total_doctor_tokens",
        "total_judge_tokens",
        "total_process_tokens",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames_summary)
    writer.writeheader()
    writer.writerow(no_memory_summary)
    writer.writerow(memory_summary)

print("\n" + "=" * 100)
print("Memory 对比实验完成")
print("=" * 100)
print("逐病例结果保存到：")
print(compare_path)
print("条件汇总结果保存到：")
print(summary_path)

print("\n关键对比：")
print(f"No Memory 平均得分：{no_memory_summary['avg_score']:.2f}")
print(f"With Memory 平均得分：{memory_summary['avg_score']:.2f}")
print(f"No Memory 平均轮数：{no_memory_summary['avg_turns']:.2f}")
print(f"With Memory 平均轮数：{memory_summary['avg_turns']:.2f}")
print(f"No Memory 平均成本：{no_memory_summary['avg_cost']:.2f}")
print(f"With Memory 平均成本：{memory_summary['avg_cost']:.2f}")
print(f"No Memory LOW_YIELD：{no_memory_summary['total_low_yield']}")
print(f"With Memory LOW_YIELD：{memory_summary['total_low_yield']}")
