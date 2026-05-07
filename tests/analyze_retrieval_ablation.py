import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

result_dir = PROJECT_ROOT / "outputs" / "retrieval_ablation"
case_summary_path = result_dir / "retrieval_ablation_case_summary.csv"
condition_summary_path = result_dir / "retrieval_ablation_condition_summary.csv"
report_path = result_dir / "retrieval_ablation_report.md"


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
    "no_memory",
    "keyword_memory",
    "embedding_memory",
]

condition_name_cn = {
    "no_memory": "No Memory",
    "keyword_memory": "Keyword Memory",
    "embedding_memory": "Embedding Memory",
}


def get_condition_row(condition_name):
    for row in condition_rows:
        if row.get("condition") == condition_name:
            return row
    raise ValueError(f"没有找到条件：{condition_name}")


def get_cases_by_condition(condition_name):
    return [row for row in case_rows if row.get("condition") == condition_name]


best_score_condition = max(
    condition_rows,
    key=lambda row: safe_float(row.get("avg_score")),
)

best_turn_condition = min(
    condition_rows,
    key=lambda row: safe_float(row.get("avg_turns")),
)

best_cost_condition = min(
    condition_rows,
    key=lambda row: safe_float(row.get("avg_cost")),
)

best_low_yield_condition = min(
    condition_rows,
    key=lambda row: safe_int(row.get("total_low_yield")),
)


report_lines = []

report_lines.append("# Mini-MedAgent Retrieval Ablation 检索方式消融实验报告")
report_lines.append("")
report_lines.append("## 1. 实验目的")
report_lines.append("")
report_lines.append("本实验比较不同 Memory 检索方式对医学诊断 Agent 的影响。")
report_lines.append("")
report_lines.append("我们设置三组实验：")
report_lines.append("")
report_lines.append("1. **No Memory**：不使用历史经验。")
report_lines.append("2. **Keyword Memory**：使用关键词重叠检索相关经验。")
report_lines.append("3. **Embedding Memory**：使用 BAAI/bge-small-zh-v1.5 生成语义向量，并基于相似度检索相关经验。")
report_lines.append("")
report_lines.append("实验目标是观察语义检索是否能比关键词检索更有效地降低交互轮数、检查成本和低价值动作。")
report_lines.append("")

report_lines.append("## 2. 条件级总体结果")
report_lines.append("")
report_lines.append("| 条件 | 平均得分 | 平均轮数 | 平均成本 | HIGH_YIELD | REASONABLE | LOW_YIELD | INEFFICIENT | CRITICAL_ERROR | Doctor tokens |")
report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for condition in condition_order:
    row = get_condition_row(condition)

    report_lines.append(
        "| {name} | {avg_score:.2f} | {avg_turns:.2f} | {avg_cost:.2f} | {high} | {reasonable} | {low} | {inefficient} | {critical} | {tokens} |".format(
            name=condition_name_cn[condition],
            avg_score=safe_float(row.get("avg_score")),
            avg_turns=safe_float(row.get("avg_turns")),
            avg_cost=safe_float(row.get("avg_cost")),
            high=safe_int(row.get("total_high_yield")),
            reasonable=safe_int(row.get("total_reasonable")),
            low=safe_int(row.get("total_low_yield")),
            inefficient=safe_int(row.get("total_inefficient")),
            critical=safe_int(row.get("total_critical_error")),
            tokens=safe_int(row.get("total_doctor_tokens")),
        )
    )

report_lines.append("")

report_lines.append("## 3. 核心发现")
report_lines.append("")
report_lines.append(f"- **最高平均得分**：{condition_name_cn[best_score_condition['condition']]}，平均得分 {safe_float(best_score_condition['avg_score']):.2f}。")
report_lines.append(f"- **最低平均轮数**：{condition_name_cn[best_turn_condition['condition']]}，平均轮数 {safe_float(best_turn_condition['avg_turns']):.2f}。")
report_lines.append(f"- **最低平均成本**：{condition_name_cn[best_cost_condition['condition']]}，平均成本 {safe_float(best_cost_condition['avg_cost']):.2f}。")
report_lines.append(f"- **LOW_YIELD 最少**：{condition_name_cn[best_low_yield_condition['condition']]}，LOW_YIELD 数量 {safe_int(best_low_yield_condition['total_low_yield'])}。")
report_lines.append("")
report_lines.append("本次实验中，三组平均 Judge 得分均为 100.00，说明三种设置都能完成最终诊断。差异主要体现在诊断效率、成本和过程质量上。")
report_lines.append("")
report_lines.append("Embedding Memory 在平均轮数、平均成本和 LOW_YIELD 数量上表现最好，但 Doctor tokens 最高，说明语义记忆检索提升了诊断路径质量，同时也增加了 prompt 注入成本。")
report_lines.append("")

report_lines.append("## 4. 相对 No Memory 的变化")
report_lines.append("")

base = get_condition_row("no_memory")
base_score = safe_float(base.get("avg_score"))
base_turns = safe_float(base.get("avg_turns"))
base_cost = safe_float(base.get("avg_cost"))
base_low = safe_int(base.get("total_low_yield"))
base_tokens = safe_int(base.get("total_doctor_tokens"))

report_lines.append("| 条件 | 得分变化 | 轮数变化 | 成本变化 | LOW_YIELD变化 | Doctor tokens变化 |")
report_lines.append("|---|---:|---:|---:|---:|---:|")

for condition in ["keyword_memory", "embedding_memory"]:
    row = get_condition_row(condition)

    score_delta = safe_float(row.get("avg_score")) - base_score
    turns_delta = safe_float(row.get("avg_turns")) - base_turns
    cost_delta = safe_float(row.get("avg_cost")) - base_cost
    low_delta = safe_int(row.get("total_low_yield")) - base_low
    token_delta = safe_int(row.get("total_doctor_tokens")) - base_tokens

    report_lines.append(
        f"| {condition_name_cn[condition]} | {score_delta:+.2f} | {turns_delta:+.2f} | {cost_delta:+.2f} | {low_delta:+d} | {token_delta:+d} |"
    )

report_lines.append("")

report_lines.append("## 5. 逐病例对比")
report_lines.append("")

case_ids = sorted(set(row["case_id"] for row in case_rows))

report_lines.append("| 病例ID | 条件 | Agent诊断 | Judge得分 | 轮数 | 成本 | LOW_YIELD | 检查项目 |")
report_lines.append("|---|---|---|---:|---:|---:|---:|---|")

for case_id in case_ids:
    for condition in condition_order:
        matched = [
            row for row in case_rows
            if row.get("case_id") == case_id and row.get("condition") == condition
        ]

        if not matched:
            continue

        row = matched[0]

        report_lines.append(
            "| {case_id} | {condition} | {diag} | {score} | {turns} | {cost} | {low} | {tests} |".format(
                case_id=case_id,
                condition=condition_name_cn[condition],
                diag=row.get("final_diagnosis", ""),
                score=row.get("score", ""),
                turns=row.get("turns", ""),
                cost=row.get("total_cost", ""),
                low=row.get("low_yield_count", ""),
                tests=row.get("ordered_tests", ""),
            )
        )

report_lines.append("")

report_lines.append("## 6. 现象解释")
report_lines.append("")
report_lines.append("### 6.1 Keyword Memory 为什么没有明显改善")
report_lines.append("")
report_lines.append("Keyword Memory 依赖主诉、历史记录和 memory 中关键词的字面重叠。")
report_lines.append("")
report_lines.append("在医学场景中，同一个疾病或诊断路径可能有多种表达方式。例如“肺栓塞”“急性肺血栓栓塞症”“突发呼吸困难伴胸痛”“D-二聚体和 CTPA”在语义上相关，但字面重叠未必充分。")
report_lines.append("")
report_lines.append("因此，Keyword Memory 可能检索到不够相关的经验，或者漏掉语义相关但字面不同的经验。它还会增加 prompt 长度，导致 token 成本上升。")
report_lines.append("")
report_lines.append("### 6.2 Embedding Memory 为什么更有效")
report_lines.append("")
report_lines.append("Embedding Memory 将当前病例上下文和历史经验转化为向量，通过语义相似度检索相关经验。")
report_lines.append("")
report_lines.append("这使得 Agent 更容易检索到医学语义相关的经验，而不仅仅依赖字面关键词匹配。")
report_lines.append("")
report_lines.append("在本实验中，Embedding Memory 将平均轮数从 5.00 降至 4.80，将平均成本从 8.80 降至 8.00，并将 LOW_YIELD 从 3 降至 2。")
report_lines.append("")
report_lines.append("### 6.3 Embedding Memory 的代价")
report_lines.append("")
report_lines.append("Embedding Memory 的 Doctor tokens 最高，说明经验注入带来了额外上下文成本。")
report_lines.append("")
report_lines.append("后续可以通过以下方式优化：")
report_lines.append("")
report_lines.append("- 减少 top_k，例如从 5 降到 3。")
report_lines.append("- 对 memory 进行压缩，只保留核心 lesson。")
report_lines.append("- 对相似度设置阈值，低相关经验不注入 prompt。")
report_lines.append("- 区分疾病场景，只注入高度相关的疾病经验。")
report_lines.append("")

report_lines.append("## 7. 局限性")
report_lines.append("")
report_lines.append("当前实验仍然是小规模初步实验，存在以下局限：")
report_lines.append("")
report_lines.append("1. 病例数量只有 5 个。")
report_lines.append("2. 病例是玩具病例，不代表真实临床复杂性。")
report_lines.append("3. JudgeAgent 和 ProcessGrader 由 LLM 扮演，可能存在评价偏差。")
report_lines.append("4. 单次运行受 LLM 随机性影响，结果需要多次重复验证。")
report_lines.append("5. Embedding Memory 当前每轮重新编码 memory，效率较低，后续应缓存 memory embeddings。")
report_lines.append("")

report_lines.append("## 8. 初步结论")
report_lines.append("")
report_lines.append("本实验完成了 Mini-MedAgent 的 Memory Retrieval Ablation Study。")
report_lines.append("")
report_lines.append("结果显示，Keyword Memory 相比 No Memory 没有带来明显收益，反而增加了平均轮数和 token 消耗；Embedding Memory 在保持诊断正确率的同时，降低了平均轮数、检查成本和 LOW_YIELD 动作数量。")
report_lines.append("")
report_lines.append("这说明在医学 Agent 中，语义向量检索比简单关键词检索更适合作为经验记忆的检索方式。")
report_lines.append("")
report_lines.append("下一步可以优化 Memory 注入策略，例如缓存 embedding、设置相似度阈值、减少 top_k，并进一步评估 memory 对诊断安全性和资源效率的影响。")

report_text = "\n".join(report_lines)

with report_path.open("w", encoding="utf-8") as f:
    f.write(report_text)

print("=" * 100)
print("Retrieval Ablation 报告生成成功")
print("=" * 100)
print("报告路径：")
print(report_path)
print("")
print("核心结果：")
for condition in condition_order:
    row = get_condition_row(condition)
    print("-" * 100)
    print(condition_name_cn[condition])
    print(f"平均得分：{safe_float(row.get('avg_score')):.2f}")
    print(f"平均轮数：{safe_float(row.get('avg_turns')):.2f}")
    print(f"平均成本：{safe_float(row.get('avg_cost')):.2f}")
    print(f"LOW_YIELD：{safe_int(row.get('total_low_yield'))}")
    print(f"Doctor tokens：{safe_int(row.get('total_doctor_tokens'))}")
