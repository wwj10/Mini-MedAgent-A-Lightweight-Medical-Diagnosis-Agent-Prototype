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
backup_path = PROJECT_ROOT / "outputs" / "memory" / "medical_memory_backup_before_embedding_config_ablation.json"

with case_path.open("r", encoding="utf-8") as f:
    cases = json.load(f)


def reset_memory():
    if memory_path.exists():
        memory_path.unlink()


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


def run_condition(condition_name, top_k, threshold):
    print("\n" + "=" * 100)
    print("开始配置：", condition_name)
    print("top_k =", top_k)
    print("threshold =", threshold)
    print("=" * 100)

    output_dir = PROJECT_ROOT / "outputs" / "embedding_config_ablation" / condition_name / "demo_transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for case in cases:
        runner = EpisodeRunner(
            case=case,
            max_turns=8,
            use_memory=True,
            memory_mode="all",
            memory_retrieval_mode="embedding",
            memory_top_k=top_k,
            memory_similarity_threshold=threshold,
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
            "memory_mode": result["memory_mode"],
            "memory_retrieval_mode": result["memory_retrieval_mode"],
            "memory_top_k": result["memory_top_k"],
            "memory_similarity_threshold": result["memory_similarity_threshold"],
            "memory_added_count": result["memory_added_count"],
            "memory_total_count": result["memory_total_count"],
            "transcript_path": str(transcript_path),
        })

    return rows


def summarize(rows, condition_name):
    n = len(rows)

    avg_score = sum(row["score"] for row in rows) / n
    avg_turns = sum(row["turns"] for row in rows) / n
    avg_cost = sum(row["total_cost"] for row in rows) / n

    total_high_yield = sum(row["high_yield_count"] for row in rows)
    total_reasonable = sum(row["reasonable_count"] for row in rows)
    total_low_yield = sum(row["low_yield_count"] for row in rows)
    total_inefficient = sum(row["inefficient_count"] for row in rows)
    total_critical_error = sum(row["critical_error_count"] for row in rows)

    total_doctor_tokens = sum(row["doctor_llm_total_tokens"] for row in rows)
    total_judge_tokens = sum(row["judge_llm_total_tokens"] for row in rows)
    total_process_tokens = sum(row["process_grader_llm_total_tokens"] for row in rows)

    return {
        "condition": condition_name,
        "num_cases": n,
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


if memory_path.exists():
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(memory_path, backup_path)
    print("已备份原 memory 到：", backup_path)


configs = [
    {
        "condition_name": "top5_threshold035",
        "top_k": 5,
        "threshold": 0.35,
    },
    {
        "condition_name": "top3_threshold040",
        "top_k": 3,
        "threshold": 0.40,
    },
    {
        "condition_name": "top2_threshold045",
        "top_k": 2,
        "threshold": 0.45,
    },
]

all_rows = []
condition_summaries = []

for config in configs:
    reset_memory()

    rows = run_condition(
        condition_name=config["condition_name"],
        top_k=config["top_k"],
        threshold=config["threshold"],
    )

    all_rows.extend(rows)

    summary = summarize(rows, config["condition_name"])
    condition_summaries.append(summary)

    print("\n" + "-" * 100)
    print(config["condition_name"], "汇总")
    print("-" * 100)
    print(f"平均得分：{summary['avg_score']:.2f}")
    print(f"平均轮数：{summary['avg_turns']:.2f}")
    print(f"平均成本：{summary['avg_cost']:.2f}")
    print(f"LOW_YIELD：{summary['total_low_yield']}")
    print(f"INEFFICIENT：{summary['total_inefficient']}")
    print(f"CRITICAL_ERROR：{summary['total_critical_error']}")
    print(f"Doctor tokens：{summary['total_doctor_tokens']}")


result_dir = PROJECT_ROOT / "outputs" / "embedding_config_ablation"
result_dir.mkdir(parents=True, exist_ok=True)

case_result_path = result_dir / "embedding_config_ablation_case_summary.csv"
condition_result_path = result_dir / "embedding_config_ablation_condition_summary.csv"

case_fieldnames = [
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
    "memory_mode",
    "memory_retrieval_mode",
    "memory_top_k",
    "memory_similarity_threshold",
    "memory_added_count",
    "memory_total_count",
    "transcript_path",
]

with case_result_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=case_fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

condition_fieldnames = [
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

with condition_result_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=condition_fieldnames)
    writer.writeheader()
    writer.writerows(condition_summaries)

print("\n" + "=" * 100)
print("Embedding Config Ablation 实验完成")
print("=" * 100)
print("逐病例结果：")
print(case_result_path)
print("条件汇总结果：")
print(condition_result_path)

print("\n最终配置对比：")
for summary in condition_summaries:
    print("-" * 100)
    print(summary["condition"])
    print(f"平均得分：{summary['avg_score']:.2f}")
    print(f"平均轮数：{summary['avg_turns']:.2f}")
    print(f"平均成本：{summary['avg_cost']:.2f}")
    print(f"LOW_YIELD：{summary['total_low_yield']}")
    print(f"INEFFICIENT：{summary['total_inefficient']}")
    print(f"CRITICAL_ERROR：{summary['total_critical_error']}")
    print(f"Doctor tokens：{summary['total_doctor_tokens']}")
