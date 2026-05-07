import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ablation_dir = PROJECT_ROOT / "outputs" / "memory_ablation"
case_summary_path = ablation_dir / "memory_ablation_case_summary.csv"
condition_summary_path = ablation_dir / "memory_ablation_condition_summary.csv"
report_path = ablation_dir / "memory_ablation_report.md"


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
    "positive_memory_only",
    "negative_memory_only",
    "all_memory",
]

condition_name_cn = {
    "no_memory": "No Memory",
    "positive_memory_only": "Positive Memory only",
    "negative_memory_only": "Negative Memory only",
    "all_memory": "All Memory",
}


def get_condition_row(condition_name):
    for row in condition_rows:
        if row.get("condition") == condition_name:
            return row
    raise ValueError(f"没有找到条件：{condition_name}")


def get_cases_by_condition(condition_name):
    return [row for row in case_rows if row.get("condition") == condition_name]


# 找最佳条件
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

report_lines.append("# Mini-MedAgent Memory Ablation 消融实验报告")
report_lines.append("")
report_lines.append("## 1. 实验目的")
report_lines.append("")
report_lines.append("本实验用于分析不同类型的经验记忆对医学诊断 Agent 行为的影响。")
report_lines.append("")
report_lines.append("我们比较四种设置：")
report_lines.append("")
report_lines.append("1. **No Memory**：不使用历史经验。")
report_lines.append("2. **Positive Memory only**：只读取 HIGH_YIELD 动作形成的正向经验。")
report_lines.append("3. **Negative Memory only**：只读取 LOW_YIELD / INEFFICIENT / CRITICAL_ERROR 动作形成的负向经验。")
report_lines.append("4. **All Memory**：同时读取正向经验和负向经验。")
report_lines.append("")
report_lines.append("实验目标是观察哪种记忆形式更有助于提升诊断准确率、降低交互轮数、减少检查成本和减少低价值动作。")
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

report_lines.append("从当前结果看，Positive Memory only 获得最高平均得分，并且没有 LOW_YIELD 动作；Negative Memory only 的平均轮数最低，也没有 LOW_YIELD 动作。All Memory 并未取得最优结果，提示正负经验混合注入时可能存在信息干扰。")
report_lines.append("")

report_lines.append("## 4. 相对 No Memory 的变化")
report_lines.append("")

base = get_condition_row("no_memory")
base_score = safe_float(base.get("avg_score"))
base_turns = safe_float(base.get("avg_turns"))
base_cost = safe_float(base.get("avg_cost"))
base_low = safe_int(base.get("total_low_yield"))

report_lines.append("| 条件 | 得分变化 | 轮数变化 | 成本变化 | LOW_YIELD变化 |")
report_lines.append("|---|---:|---:|---:|---:|")

for condition in ["positive_memory_only", "negative_memory_only", "all_memory"]:
    row = get_condition_row(condition)

    score_delta = safe_float(row.get("avg_score")) - base_score
    turns_delta = safe_float(row.get("avg_turns")) - base_turns
    cost_delta = safe_float(row.get("avg_cost")) - base_cost
    low_delta = safe_int(row.get("total_low_yield")) - base_low

    report_lines.append(
        f"| {condition_name_cn[condition]} | {score_delta:+.2f} | {turns_delta:+.2f} | {cost_delta:+.2f} | {low_delta:+d} |"
    )

report_lines.append("")

report_lines.append("## 5. 逐病例对比")
report_lines.append("")

# 构造 case_id 集合
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
report_lines.append("### 6.1 Positive Memory only 为什么效果好")
report_lines.append("")
report_lines.append("Positive Memory only 只注入 HIGH_YIELD 动作经验，因此它主要告诉 DoctorAgent：在相似病例中哪些问诊或检查值得优先做。")
report_lines.append("")
report_lines.append("这种记忆形式更像“临床路径提示”，可以帮助 Agent 更快选择高收益动作，因此本实验中它取得了最高平均得分，并将 LOW_YIELD 降为 0。")
report_lines.append("")
report_lines.append("### 6.2 Negative Memory only 为什么能减少轮数")
report_lines.append("")
report_lines.append("Negative Memory only 主要告诉 Agent 哪些动作应避免，例如低价值问诊或不必要检查。")
report_lines.append("")
report_lines.append("这种记忆形式可能让 Agent 更快跳过低价值步骤，因此平均轮数最低。但它不直接告诉 Agent 最优动作是什么，因此得分没有超过 Positive Memory only。")
report_lines.append("")
report_lines.append("### 6.3 All Memory 为什么没有最好")
report_lines.append("")
report_lines.append("All Memory 同时注入正向和负向经验，理论上信息最完整，但在小模型和短 prompt 场景下，混合经验可能造成注意力分散。")
report_lines.append("")
report_lines.append("例如，正向经验鼓励某些检查，负向经验又提示避免某些低价值动作，两者同时出现时可能增加决策负担，导致 All Memory 没有超过单独的 Positive 或 Negative Memory。")
report_lines.append("")

report_lines.append("## 7. 局限性")
report_lines.append("")
report_lines.append("当前消融实验仍然是初步实验，不能得出强结论，主要局限包括：")
report_lines.append("")
report_lines.append("1. 病例数量只有 5 个，样本过小。")
report_lines.append("2. 病例是玩具病例，不能代表真实临床复杂性。")
report_lines.append("3. PatientAgent 和 ExamAgent 是规则环境，不能模拟真实患者表达差异。")
report_lines.append("4. Memory 检索仍然是关键词匹配，不是 embedding 检索。")
report_lines.append("5. LLM 输出存在随机性，单次运行结果可能波动。")
report_lines.append("6. ProcessGrader 和 JudgeAgent 也由 LLM 扮演，可能存在评价偏差。")
report_lines.append("")

report_lines.append("## 8. 初步结论")
report_lines.append("")
report_lines.append("本实验完成了 Mini-MedAgent 的 Memory Ablation Study。结果显示，经验记忆能够改变 Agent 的诊断行为。")
report_lines.append("")
report_lines.append("在当前 5 个玩具病例上，Positive Memory only 获得最高平均得分，Negative Memory only 获得最低平均轮数，而 All Memory 并未表现最好。")
report_lines.append("")
report_lines.append("这说明不同类型的经验对 Agent 行为有不同影响：正向经验更有助于复用高价值诊断路径，负向经验更有助于减少低价值动作，而简单混合两类经验可能带来信息干扰。")
report_lines.append("")
report_lines.append("下一步可以将 MemoryManager 从关键词检索升级为 embedding 检索，并增加更多病例，以验证该结论是否稳定。")

report_text = "\n".join(report_lines)

with report_path.open("w", encoding="utf-8") as f:
    f.write(report_text)

print("=" * 100)
print("Memory Ablation 报告生成成功")
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
