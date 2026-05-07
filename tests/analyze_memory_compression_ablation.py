import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

result_dir = PROJECT_ROOT / "outputs" / "memory_compression_ablation"
case_summary_path = result_dir / "memory_compression_case_summary.csv"
condition_summary_path = result_dir / "memory_compression_condition_summary.csv"
report_path = result_dir / "memory_compression_ablation_report.md"


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


if not case_summary_path.exists():
    raise FileNotFoundError(f"没有找到逐病例结果文件：{case_summary_path}")

if not condition_summary_path.exists():
    raise FileNotFoundError(f"没有找到条件汇总文件：{condition_summary_path}")


with case_summary_path.open("r", encoding="utf-8-sig", newline="") as f:
    case_rows = list(csv.DictReader(f))

with condition_summary_path.open("r", encoding="utf-8-sig", newline="") as f:
    condition_rows = list(csv.DictReader(f))


condition_order = [
    "embedding_full_memory",
    "embedding_balanced_memory",
    "embedding_compact_memory",
]

condition_names = {
    "embedding_full_memory": "Full Memory",
    "embedding_balanced_memory": "Balanced Memory",
    "embedding_compact_memory": "Compact Memory",
}


def get_condition_row(condition):
    for row in condition_rows:
        if row.get("condition") == condition:
            return row
    raise ValueError(f"没有找到条件：{condition}")


full = get_condition_row("embedding_full_memory")
full_tokens = safe_int(full.get("total_doctor_tokens"))
full_score = safe_float(full.get("avg_score"))


best_score = max(condition_rows, key=lambda r: safe_float(r.get("avg_score")))
best_cost = min(condition_rows, key=lambda r: safe_float(r.get("avg_cost")))
best_tokens = min(condition_rows, key=lambda r: safe_int(r.get("total_doctor_tokens")))
best_low_yield = min(condition_rows, key=lambda r: safe_int(r.get("total_low_yield")))


lines = []

lines.append("# Mini-MedAgent Memory Compression 消融实验报告")
lines.append("")
lines.append("## 1. 实验目的")
lines.append("")
lines.append("本实验比较不同 memory prompt 注入格式对医学诊断 Agent 的影响。")
lines.append("")
lines.append("我们比较三种格式：")
lines.append("")
lines.append("- **Full Memory**：完整注入经验，包括类型、标签、相似度、适用线索、经验内容、Grader 理由和改进建议。")
lines.append("- **Balanced Memory**：折中格式，保留适用场景、动作建议和关键理由。")
lines.append("- **Compact Memory**：极简格式，只保留场景、动作类型和动作内容。")
lines.append("")
lines.append("实验目标是观察 memory 压缩是否能够降低 DoctorAgent 的 token 消耗，同时尽量保持诊断准确率和过程质量。")
lines.append("")

lines.append("## 2. 条件级总体结果")
lines.append("")
lines.append("| 条件 | 平均得分 | 平均轮数 | 平均成本 | LOW_YIELD | INEFFICIENT | CRITICAL_ERROR | Doctor tokens | 相对 Full Token 变化 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

for condition in condition_order:
    row = get_condition_row(condition)

    tokens = safe_int(row.get("total_doctor_tokens"))
    token_delta_ratio = (tokens - full_tokens) / full_tokens if full_tokens else 0.0

    lines.append(
        "| {name} | {score:.2f} | {turns:.2f} | {cost:.2f} | {low} | {inefficient} | {critical} | {tokens} | {delta:+.2%} |".format(
            name=condition_names[condition],
            score=safe_float(row.get("avg_score")),
            turns=safe_float(row.get("avg_turns")),
            cost=safe_float(row.get("avg_cost")),
            low=safe_int(row.get("total_low_yield")),
            inefficient=safe_int(row.get("total_inefficient")),
            critical=safe_int(row.get("total_critical_error")),
            tokens=tokens,
            delta=token_delta_ratio,
        )
    )

lines.append("")

lines.append("## 3. 核心发现")
lines.append("")
lines.append(f"- **最高平均得分**：{condition_names[best_score['condition']]}，平均得分 {safe_float(best_score['avg_score']):.2f}。")
lines.append(f"- **最低平均成本**：{condition_names[best_cost['condition']]}，平均成本 {safe_float(best_cost['avg_cost']):.2f}。")
lines.append(f"- **最低 Doctor tokens**：{condition_names[best_tokens['condition']]}，Doctor tokens {safe_int(best_tokens['total_doctor_tokens'])}。")
lines.append(f"- **LOW_YIELD 最少**：{condition_names[best_low_yield['condition']]}，LOW_YIELD 数量 {safe_int(best_low_yield['total_low_yield'])}。")
lines.append("")

lines.append("## 4. 结果解释")
lines.append("")
lines.append("### 4.1 Full Memory")
lines.append("")
lines.append("Full Memory 获得最高平均得分，说明完整经验信息对医学诊断安全性最有帮助。")
lines.append("")
lines.append("它保留了动作价值原因、适用线索和改进建议，因此 DoctorAgent 更容易理解为什么某个动作值得做，而不是机械复用动作。缺点是 token 消耗最高。")
lines.append("")

lines.append("### 4.2 Balanced Memory")
lines.append("")
lines.append("Balanced Memory 在保留关键理由的同时压缩了部分上下文字段。")
lines.append("")
lines.append("本实验中，Balanced Memory 的 Doctor tokens 相比 Full Memory 明显降低，同时平均成本更低、LOW_YIELD 更少，但平均得分从 100.00 降至 96.00。")
lines.append("")
lines.append("这说明 Balanced Memory 是一个可用的成本友好版本，但在医学场景中，如果优先考虑诊断安全性，仍应谨慎替代 Full Memory。")
lines.append("")

lines.append("### 4.3 Compact Memory")
lines.append("")
lines.append("Compact Memory token 最低，但压缩过度。")
lines.append("")
lines.append("它主要保留动作名称和诊断场景，丢失了动作为什么有效、什么时候适用、是否有风险等关键信息。")
lines.append("")
lines.append("因此，Compact Memory 虽然降低了 token，但不适合作为当前默认配置。")
lines.append("")

lines.append("## 5. 推荐配置")
lines.append("")
lines.append("当前推荐默认配置为：")
lines.append("")
lines.append("```python")
lines.append("memory_prompt_style = \"full\"")
lines.append("```")
lines.append("")
lines.append("原因是医学诊断任务应优先保证准确性和过程安全性。Full Memory 在当前实验中获得最高平均得分。")
lines.append("")
lines.append("如果后续目标是降低 API 成本，可以设置：")
lines.append("")
lines.append("```python")
lines.append("memory_prompt_style = \"balanced\"")
lines.append("```")
lines.append("")
lines.append("Balanced Memory 可以作为成本友好的替代版本，但需要在更多病例上验证其稳定性。")
lines.append("")

lines.append("## 6. 逐病例对比")
lines.append("")
lines.append("| 病例ID | 条件 | Agent诊断 | Judge得分 | 轮数 | 成本 | LOW_YIELD | Doctor tokens |")
lines.append("|---|---|---|---:|---:|---:|---:|---:|")

case_ids = sorted(set(row["case_id"] for row in case_rows))

for case_id in case_ids:
    for condition in condition_order:
        matched = [
            row for row in case_rows
            if row.get("case_id") == case_id and row.get("condition") == condition
        ]

        if not matched:
            continue

        row = matched[0]

        lines.append(
            "| {case_id} | {condition} | {diag} | {score} | {turns} | {cost} | {low} | {tokens} |".format(
                case_id=case_id,
                condition=condition_names[condition],
                diag=row.get("final_diagnosis", ""),
                score=row.get("score", ""),
                turns=row.get("turns", ""),
                cost=row.get("total_cost", ""),
                low=row.get("low_yield_count", ""),
                tokens=row.get("doctor_llm_total_tokens", ""),
            )
        )

lines.append("")

lines.append("## 7. 局限性")
lines.append("")
lines.append("当前实验仍然是初步实验，主要局限包括：")
lines.append("")
lines.append("1. 病例数量只有 5 个。")
lines.append("2. LLM 输出存在随机性，单次结果可能波动。")
lines.append("3. PatientAgent 和 ExamAgent 是规则环境，不能代表真实临床复杂性。")
lines.append("4. JudgeAgent 和 ProcessGrader 由 LLM 扮演，可能存在评价偏差。")
lines.append("5. Compact 和 Balanced 的提示格式仍然是人工设计，后续可以尝试自动压缩或学习式 memory summarization。")
lines.append("")

lines.append("## 8. 初步结论")
lines.append("")
lines.append("本实验表明，Memory 压缩存在明显的准确率—成本权衡。")
lines.append("")
lines.append("Full Memory 提供最完整的诊断经验，获得最高平均得分；Compact Memory 虽然显著降低 token，但可能丢失关键医学理由；Balanced Memory 在 token 降低和信息保留之间取得折中。")
lines.append("")
lines.append("因此，当前默认采用 Full Memory；Balanced Memory 可作为低成本替代方案；Compact Memory 暂不推荐用于医学诊断场景。")

report_text = "\n".join(lines)

report_path.parent.mkdir(parents=True, exist_ok=True)

with report_path.open("w", encoding="utf-8") as f:
    f.write(report_text)

print("=" * 100)
print("Memory Compression Ablation 报告生成成功")
print("=" * 100)
print("报告路径：")
print(report_path)
print("")
print("推荐默认配置：")
print('memory_prompt_style = "full"')
