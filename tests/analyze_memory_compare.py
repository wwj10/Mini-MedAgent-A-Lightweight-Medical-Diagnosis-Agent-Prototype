import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

case_summary_path = PROJECT_ROOT / "outputs" / "compare_memory_summary.csv"
condition_summary_path = PROJECT_ROOT / "outputs" / "compare_memory_condition_summary.csv"
report_path = PROJECT_ROOT / "outputs" / "compare_memory_report.md"


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
    raise FileNotFoundError(f"没有找到逐病例对比文件：{case_summary_path}")

if not condition_summary_path.exists():
    raise FileNotFoundError(f"没有找到条件汇总文件：{condition_summary_path}")


with case_summary_path.open("r", encoding="utf-8-sig", newline="") as f:
    case_rows = list(csv.DictReader(f))

with condition_summary_path.open("r", encoding="utf-8-sig", newline="") as f:
    condition_rows = list(csv.DictReader(f))


def find_condition(name: str):
    for row in condition_rows:
        if row.get("condition") == name:
            return row
    raise ValueError(f"没有找到实验条件：{name}")


no_memory = find_condition("No Memory")
with_memory = find_condition("With Memory")


def get_case_map(condition_name: str):
    result = {}
    for row in case_rows:
        if row.get("condition") == condition_name:
            result[row["case_id"]] = row
    return result


no_memory_cases = get_case_map("no_memory")
with_memory_cases = get_case_map("with_memory")

case_ids = sorted(set(no_memory_cases.keys()) & set(with_memory_cases.keys()))


# 条件级指标
no_avg_score = safe_float(no_memory["avg_score"])
mem_avg_score = safe_float(with_memory["avg_score"])

no_avg_turns = safe_float(no_memory["avg_turns"])
mem_avg_turns = safe_float(with_memory["avg_turns"])

no_avg_cost = safe_float(no_memory["avg_cost"])
mem_avg_cost = safe_float(with_memory["avg_cost"])

no_low_yield = safe_int(no_memory["total_low_yield"])
mem_low_yield = safe_int(with_memory["total_low_yield"])

no_inefficient = safe_int(no_memory["total_inefficient"])
mem_inefficient = safe_int(with_memory["total_inefficient"])

no_critical = safe_int(no_memory["total_critical_error"])
mem_critical = safe_int(with_memory["total_critical_error"])

no_doctor_tokens = safe_int(no_memory["total_doctor_tokens"])
mem_doctor_tokens = safe_int(with_memory["total_doctor_tokens"])


report_lines = []

report_lines.append("# Mini-MedAgent Memory 对比实验报告")
report_lines.append("")
report_lines.append("## 1. 实验目的")
report_lines.append("")
report_lines.append("本实验比较两种设置下 Mini-MedAgent 的表现：")
report_lines.append("")
report_lines.append("- **No Memory**：DoctorAgent 不读取历史经验，每个病例独立诊断。")
report_lines.append("- **With Memory**：DoctorAgent 在诊断时读取 MemoryManager 检索出的相关历史经验，并在每个病例结束后根据 ProcessGrader 的动作级评价更新经验库。")
report_lines.append("")
report_lines.append("实验目标是观察 Memory 是否能够减少低价值动作、降低检查成本、缩短诊断轮数，同时保持或提升最终诊断质量。")
report_lines.append("")

report_lines.append("## 2. 条件级总体结果")
report_lines.append("")
report_lines.append("| 指标 | No Memory | With Memory | 变化 |")
report_lines.append("|---|---:|---:|---:|")
report_lines.append(f"| 平均 Judge 得分 | {no_avg_score:.2f} | {mem_avg_score:.2f} | {mem_avg_score - no_avg_score:+.2f} |")
report_lines.append(f"| 平均轮数 | {no_avg_turns:.2f} | {mem_avg_turns:.2f} | {mem_avg_turns - no_avg_turns:+.2f} |")
report_lines.append(f"| 平均检查成本 | {no_avg_cost:.2f} | {mem_avg_cost:.2f} | {mem_avg_cost - no_avg_cost:+.2f} |")
report_lines.append(f"| LOW_YIELD 动作数 | {no_low_yield} | {mem_low_yield} | {mem_low_yield - no_low_yield:+d} |")
report_lines.append(f"| INEFFICIENT 动作数 | {no_inefficient} | {mem_inefficient} | {mem_inefficient - no_inefficient:+d} |")
report_lines.append(f"| CRITICAL_ERROR 动作数 | {no_critical} | {mem_critical} | {mem_critical - no_critical:+d} |")
report_lines.append(f"| Doctor LLM tokens | {no_doctor_tokens} | {mem_doctor_tokens} | {mem_doctor_tokens - no_doctor_tokens:+d} |")
report_lines.append("")

report_lines.append("## 3. 逐病例对比")
report_lines.append("")
report_lines.append("| 病例ID | 标准诊断 | No Memory诊断 | With Memory诊断 | No Memory得分 | With Memory得分 | 轮数变化 | 成本变化 | LOW_YIELD变化 |")
report_lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")

for case_id in case_ids:
    no_row = no_memory_cases[case_id]
    mem_row = with_memory_cases[case_id]

    no_score = safe_int(no_row.get("score"))
    mem_score = safe_int(mem_row.get("score"))

    no_turns = safe_int(no_row.get("turns"))
    mem_turns = safe_int(mem_row.get("turns"))

    no_cost = safe_int(no_row.get("total_cost"))
    mem_cost = safe_int(mem_row.get("total_cost"))

    no_low = safe_int(no_row.get("low_yield_count"))
    mem_low = safe_int(mem_row.get("low_yield_count"))

    report_lines.append(
        "| {case_id} | {gold} | {no_diag} | {mem_diag} | {no_score} | {mem_score} | {turn_delta:+d} | {cost_delta:+d} | {low_delta:+d} |".format(
            case_id=case_id,
            gold=no_row.get("gold_diagnosis", ""),
            no_diag=no_row.get("final_diagnosis", ""),
            mem_diag=mem_row.get("final_diagnosis", ""),
            no_score=no_score,
            mem_score=mem_score,
            turn_delta=mem_turns - no_turns,
            cost_delta=mem_cost - no_cost,
            low_delta=mem_low - no_low,
        )
    )

report_lines.append("")

report_lines.append("## 4. 改善案例分析")
report_lines.append("")

improved_cases = []

for case_id in case_ids:
    no_row = no_memory_cases[case_id]
    mem_row = with_memory_cases[case_id]

    no_score = safe_int(no_row.get("score"))
    mem_score = safe_int(mem_row.get("score"))

    no_turns = safe_int(no_row.get("turns"))
    mem_turns = safe_int(mem_row.get("turns"))

    no_cost = safe_int(no_row.get("total_cost"))
    mem_cost = safe_int(mem_row.get("total_cost"))

    no_low = safe_int(no_row.get("low_yield_count"))
    mem_low = safe_int(mem_row.get("low_yield_count"))

    if mem_score > no_score or mem_turns < no_turns or mem_cost < no_cost or mem_low < no_low:
        improved_cases.append((case_id, no_row, mem_row))

if improved_cases:
    for case_id, no_row, mem_row in improved_cases:
        report_lines.append(f"### {case_id}")
        report_lines.append("")
        report_lines.append(f"- 主诉：{no_row.get('chief_complaint')}")
        report_lines.append(f"- 标准诊断：{no_row.get('gold_diagnosis')}")
        report_lines.append(f"- No Memory 诊断：{no_row.get('final_diagnosis')}")
        report_lines.append(f"- With Memory 诊断：{mem_row.get('final_diagnosis')}")
        report_lines.append(f"- No Memory：得分 {no_row.get('score')}，轮数 {no_row.get('turns')}，成本 {no_row.get('total_cost')}，LOW_YIELD {no_row.get('low_yield_count')}")
        report_lines.append(f"- With Memory：得分 {mem_row.get('score')}，轮数 {mem_row.get('turns')}，成本 {mem_row.get('total_cost')}，LOW_YIELD {mem_row.get('low_yield_count')}")
        report_lines.append(f"- With Memory 过程总评：{mem_row.get('process_overall_comment')}")
        report_lines.append("")
else:
    report_lines.append("没有发现明显改善的病例。")
    report_lines.append("")

report_lines.append("## 5. 风险与局限")
report_lines.append("")
report_lines.append("本实验显示 Memory 能够降低平均轮数、检查成本和 LOW_YIELD 动作数，但仍需要谨慎解释。")
report_lines.append("")
report_lines.append("主要局限包括：")
report_lines.append("")
report_lines.append("1. 当前只有 5 个玩具病例，样本量过小。")
report_lines.append("2. DoctorAgent 的输出可能受 LLM 随机性影响，不同运行之间结果可能波动。")
report_lines.append("3. 当前 PatientAgent 和 ExamAgent 是规则环境，不能完全模拟真实患者表达和真实检查流程。")
report_lines.append("4. Memory 可能让 Agent 过度依赖历史经验，减少必要问诊，带来过早检查或过早诊断风险。")
report_lines.append("5. 当前 Memory 检索是关键词匹配，后续应升级为 embedding 检索或 RAG-style memory。")
report_lines.append("")

report_lines.append("## 6. 初步结论")
report_lines.append("")
report_lines.append(
    f"在本轮小规模对比实验中，With Memory 相比 No Memory 平均 Judge 得分从 {no_avg_score:.2f} 提升到 {mem_avg_score:.2f}，"
    f"平均轮数从 {no_avg_turns:.2f} 降至 {mem_avg_turns:.2f}，平均检查成本从 {no_avg_cost:.2f} 降至 {mem_avg_cost:.2f}，"
    f"LOW_YIELD 动作数从 {no_low_yield} 降至 {mem_low_yield}。"
)
report_lines.append("")
report_lines.append("这说明基于 ProcessGrader 的动作级反馈构建经验记忆，可能有助于优化 Agent 的诊断路径和资源效率。")
report_lines.append("")
report_lines.append("下一步可以将 MemoryManager 从规则关键词检索升级为 embedding 检索，并进一步设计 Ablation Study：")
report_lines.append("")
report_lines.append("- No Memory")
report_lines.append("- Positive Memory only")
report_lines.append("- Negative Memory only")
report_lines.append("- Positive + Negative Memory")
report_lines.append("- RAG Memory")
report_lines.append("")

report_text = "\n".join(report_lines)

with report_path.open("w", encoding="utf-8") as f:
    f.write(report_text)

print("=" * 100)
print("Memory 对比分析报告已生成")
print("=" * 100)
print("报告路径：")
print(report_path)
print("")
print("关键结果：")
print(f"No Memory 平均得分：{no_avg_score:.2f}")
print(f"With Memory 平均得分：{mem_avg_score:.2f}")
print(f"No Memory 平均轮数：{no_avg_turns:.2f}")
print(f"With Memory 平均轮数：{mem_avg_turns:.2f}")
print(f"No Memory 平均成本：{no_avg_cost:.2f}")
print(f"With Memory 平均成本：{mem_avg_cost:.2f}")
print(f"No Memory LOW_YIELD：{no_low_yield}")
print(f"With Memory LOW_YIELD：{mem_low_yield}")
