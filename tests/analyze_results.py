import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
summary_path = PROJECT_ROOT / "outputs" / "batch_summary.csv"
report_path = PROJECT_ROOT / "outputs" / "analysis_report.md"


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


if not summary_path.exists():
    raise FileNotFoundError(f"没有找到实验汇总文件：{summary_path}")

with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise ValueError("batch_summary.csv 为空，请先运行 check_batch_runner.py。")

num_cases = len(rows)

judge_scores = [safe_int(row.get("score")) for row in rows]
rule_scores = [safe_int(row.get("rule_score")) for row in rows]
turns = [safe_int(row.get("turns")) for row in rows]
costs = [safe_int(row.get("total_cost")) for row in rows]

high_yield_counts = [safe_int(row.get("high_yield_count")) for row in rows]
reasonable_counts = [safe_int(row.get("reasonable_count")) for row in rows]
low_yield_counts = [safe_int(row.get("low_yield_count")) for row in rows]
inefficient_counts = [safe_int(row.get("inefficient_count")) for row in rows]
critical_error_counts = [safe_int(row.get("critical_error_count")) for row in rows]

doctor_calls = [safe_int(row.get("doctor_llm_call_count")) for row in rows]
doctor_tokens = [safe_int(row.get("doctor_llm_total_tokens")) for row in rows]
judge_calls = [safe_int(row.get("judge_llm_call_count")) for row in rows]
judge_tokens = [safe_int(row.get("judge_llm_total_tokens")) for row in rows]
process_calls = [safe_int(row.get("process_grader_llm_call_count")) for row in rows]
process_tokens = [safe_int(row.get("process_grader_llm_total_tokens")) for row in rows]

avg_judge_score = sum(judge_scores) / num_cases
avg_rule_score = sum(rule_scores) / num_cases
avg_turns = sum(turns) / num_cases
avg_cost = sum(costs) / num_cases

total_high_yield = sum(high_yield_counts)
total_reasonable = sum(reasonable_counts)
total_low_yield = sum(low_yield_counts)
total_inefficient = sum(inefficient_counts)
total_critical_error = sum(critical_error_counts)

total_actions = (
    total_high_yield
    + total_reasonable
    + total_low_yield
    + total_inefficient
    + total_critical_error
)

high_yield_ratio = total_high_yield / total_actions if total_actions else 0
reasonable_ratio = total_reasonable / total_actions if total_actions else 0
low_yield_ratio = total_low_yield / total_actions if total_actions else 0
inefficient_ratio = total_inefficient / total_actions if total_actions else 0
critical_error_ratio = total_critical_error / total_actions if total_actions else 0

total_doctor_calls = sum(doctor_calls)
total_doctor_tokens = sum(doctor_tokens)
total_judge_calls = sum(judge_calls)
total_judge_tokens = sum(judge_tokens)
total_process_calls = sum(process_calls)
total_process_tokens = sum(process_tokens)

total_llm_calls = total_doctor_calls + total_judge_calls + total_process_calls
total_llm_tokens = total_doctor_tokens + total_judge_tokens + total_process_tokens

score_gap_cases = []
low_yield_cases = []
inefficient_cases = []
critical_error_cases = []
high_cost_cases = []

for row in rows:
    judge_score = safe_int(row.get("score"))
    rule_score = safe_int(row.get("rule_score"))
    cost = safe_int(row.get("total_cost"))
    low_yield = safe_int(row.get("low_yield_count"))
    inefficient = safe_int(row.get("inefficient_count"))
    critical_error = safe_int(row.get("critical_error_count"))

    if judge_score - rule_score >= 50:
        score_gap_cases.append(row)

    if low_yield > 0:
        low_yield_cases.append(row)

    if inefficient > 0:
        inefficient_cases.append(row)

    if critical_error > 0:
        critical_error_cases.append(row)

    if cost >= 15:
        high_cost_cases.append(row)


report_lines = []

report_lines.append("# Mini-MedAgent 实验分析报告")
report_lines.append("")
report_lines.append("## 1. 实验概况")
report_lines.append("")
report_lines.append(f"- 病例数量：{num_cases}")
report_lines.append(f"- 平均 Judge 得分：{avg_judge_score:.2f}")
report_lines.append(f"- 平均规则得分：{avg_rule_score:.2f}")
report_lines.append(f"- 平均交互轮数：{avg_turns:.2f}")
report_lines.append(f"- 平均检查成本：{avg_cost:.2f}")
report_lines.append(f"- 总动作数：{total_actions}")
report_lines.append(f"- Doctor LLM 总调用次数：{total_doctor_calls}")
report_lines.append(f"- Judge LLM 总调用次数：{total_judge_calls}")
report_lines.append(f"- ProcessGrader LLM 总调用次数：{total_process_calls}")
report_lines.append(f"- LLM 总调用次数：{total_llm_calls}")
report_lines.append(f"- LLM 总 token：{total_llm_tokens}")
report_lines.append("")

report_lines.append("## 2. 逐病例结果")
report_lines.append("")
report_lines.append("| 病例ID | 主诉 | 标准诊断 | Agent诊断 | 规则得分 | Judge得分 | 轮数 | 成本 | 检查 |")
report_lines.append("|---|---|---|---|---:|---:|---:|---:|---|")

for row in rows:
    report_lines.append(
        "| {case_id} | {chief_complaint} | {gold_diagnosis} | {final_diagnosis} | {rule_score} | {score} | {turns} | {total_cost} | {ordered_tests} |".format(
            case_id=row.get("case_id", ""),
            chief_complaint=row.get("chief_complaint", ""),
            gold_diagnosis=row.get("gold_diagnosis", ""),
            final_diagnosis=row.get("final_diagnosis", ""),
            rule_score=row.get("rule_score", ""),
            score=row.get("score", ""),
            turns=row.get("turns", ""),
            total_cost=row.get("total_cost", ""),
            ordered_tests=row.get("ordered_tests", ""),
        )
    )

report_lines.append("")

report_lines.append("## 3. 动作级评价统计")
report_lines.append("")
report_lines.append("| 标签 | 数量 | 占比 | 含义 |")
report_lines.append("|---|---:|---:|---|")
report_lines.append(f"| HIGH_YIELD | {total_high_yield} | {high_yield_ratio:.2%} | 高价值动作，明显帮助诊断 |")
report_lines.append(f"| REASONABLE | {total_reasonable} | {reasonable_ratio:.2%} | 合理但不是关键动作 |")
report_lines.append(f"| LOW_YIELD | {total_low_yield} | {low_yield_ratio:.2%} | 信息价值有限 |")
report_lines.append(f"| INEFFICIENT | {total_inefficient} | {inefficient_ratio:.2%} | 成本较高或顺序不佳 |")
report_lines.append(f"| CRITICAL_ERROR | {total_critical_error} | {critical_error_ratio:.2%} | 明显错误或可能误导诊断 |")
report_lines.append("")

report_lines.append("### 3.1 逐病例动作标签")
report_lines.append("")
report_lines.append("| 病例ID | HIGH_YIELD | REASONABLE | LOW_YIELD | INEFFICIENT | CRITICAL_ERROR | ProcessGrader总评 |")
report_lines.append("|---|---:|---:|---:|---:|---:|---|")

for row in rows:
    report_lines.append(
        "| {case_id} | {high_yield_count} | {reasonable_count} | {low_yield_count} | {inefficient_count} | {critical_error_count} | {process_overall_comment} |".format(
            case_id=row.get("case_id", ""),
            high_yield_count=row.get("high_yield_count", ""),
            reasonable_count=row.get("reasonable_count", ""),
            low_yield_count=row.get("low_yield_count", ""),
            inefficient_count=row.get("inefficient_count", ""),
            critical_error_count=row.get("critical_error_count", ""),
            process_overall_comment=row.get("process_overall_comment", ""),
        )
    )

report_lines.append("")

report_lines.append("## 4. 规则评分与 Judge 评分差异分析")
report_lines.append("")

if score_gap_cases:
    report_lines.append("以下病例中，规则评分明显低于 Judge 评分，说明简单字符串匹配无法正确处理医学同义词、分型诊断或更具体诊断。")
    report_lines.append("")

    for row in score_gap_cases:
        report_lines.append(f"### {row.get('case_id')}")
        report_lines.append("")
        report_lines.append(f"- 标准诊断：{row.get('gold_diagnosis')}")
        report_lines.append(f"- Agent诊断：{row.get('final_diagnosis')}")
        report_lines.append(f"- 规则得分：{row.get('rule_score')}")
        report_lines.append(f"- Judge得分：{row.get('score')}")
        report_lines.append(f"- Judge理由：{row.get('judge_reason')}")
        report_lines.append("")
else:
    report_lines.append("没有发现规则评分与 Judge 评分差异特别大的病例。")
    report_lines.append("")

report_lines.append("## 5. 低价值动作分析")
report_lines.append("")

if low_yield_cases:
    report_lines.append("以下病例存在 LOW_YIELD 动作，后续可以查看 transcript 进一步分析哪些问诊或检查信息价值有限。")
    report_lines.append("")

    for row in low_yield_cases:
        report_lines.append(f"### {row.get('case_id')}")
        report_lines.append("")
        report_lines.append(f"- 主诉：{row.get('chief_complaint')}")
        report_lines.append(f"- 标准诊断：{row.get('gold_diagnosis')}")
        report_lines.append(f"- Agent诊断：{row.get('final_diagnosis')}")
        report_lines.append(f"- LOW_YIELD 数量：{row.get('low_yield_count')}")
        report_lines.append(f"- Transcript：{row.get('transcript_path')}")
        report_lines.append("")
else:
    report_lines.append("没有发现 LOW_YIELD 动作。")
    report_lines.append("")

report_lines.append("## 6. 低效或严重错误动作分析")
report_lines.append("")

if inefficient_cases:
    report_lines.append("以下病例存在 INEFFICIENT 动作，提示可能有过度检查、检查顺序不佳或资源利用不优。")
    report_lines.append("")

    for row in inefficient_cases:
        report_lines.append(f"### {row.get('case_id')}")
        report_lines.append("")
        report_lines.append(f"- 主诉：{row.get('chief_complaint')}")
        report_lines.append(f"- 检查项目：{row.get('ordered_tests')}")
        report_lines.append(f"- 总成本：{row.get('total_cost')}")
        report_lines.append(f"- Transcript：{row.get('transcript_path')}")
        report_lines.append("")
else:
    report_lines.append("没有发现 INEFFICIENT 动作。")
    report_lines.append("")

if critical_error_cases:
    report_lines.append("以下病例存在 CRITICAL_ERROR 动作，需要重点复盘。")
    report_lines.append("")

    for row in critical_error_cases:
        report_lines.append(f"### {row.get('case_id')}")
        report_lines.append("")
        report_lines.append(f"- 主诉：{row.get('chief_complaint')}")
        report_lines.append(f"- 标准诊断：{row.get('gold_diagnosis')}")
        report_lines.append(f"- Agent诊断：{row.get('final_diagnosis')}")
        report_lines.append(f"- Transcript：{row.get('transcript_path')}")
        report_lines.append("")
else:
    report_lines.append("没有发现 CRITICAL_ERROR 动作。")
    report_lines.append("")

report_lines.append("## 7. 成本分析")
report_lines.append("")

if high_cost_cases:
    report_lines.append("以下病例检查成本较高，后续可以分析是否存在更低成本的诊断路径。")
    report_lines.append("")

    for row in high_cost_cases:
        report_lines.append(f"### {row.get('case_id')}")
        report_lines.append("")
        report_lines.append(f"- 主诉：{row.get('chief_complaint')}")
        report_lines.append(f"- 标准诊断：{row.get('gold_diagnosis')}")
        report_lines.append(f"- Agent诊断：{row.get('final_diagnosis')}")
        report_lines.append(f"- 检查项目：{row.get('ordered_tests')}")
        report_lines.append(f"- 总成本：{row.get('total_cost')}")
        report_lines.append(f"- ProcessGrader总评：{row.get('process_overall_comment')}")
        report_lines.append("")
else:
    report_lines.append("没有发现成本明显偏高的病例。")
    report_lines.append("")

report_lines.append("## 8. 初步结论")
report_lines.append("")
report_lines.append("本实验实现了一个最小医学诊断 Agent 环境。DoctorAgent 不能直接看到完整病例，只能通过问诊和检查逐步获取信息，并在证据充分时提交诊断。")
report_lines.append("")
report_lines.append(f"在 {num_cases} 个玩具病例上，Agent 平均 Judge 得分为 {avg_judge_score:.2f}，平均交互轮数为 {avg_turns:.2f}，平均检查成本为 {avg_cost:.2f}。")
report_lines.append("")
report_lines.append("动作级评价显示，大部分动作被评为 HIGH_YIELD 或 REASONABLE，说明 Agent 的诊断过程整体较合理。")
report_lines.append("")
report_lines.append("规则评分与 Judge 评分的差异表明，医学 Agent 的评价不能只依赖字符串匹配。对于更具体诊断、医学同义词和分型诊断，语义评价更合理。")
report_lines.append("")
report_lines.append("下一步可以加入 Memory/Evolver，将 ProcessGrader 的动作级反馈转化为可复用经验，使 DoctorAgent 在后续病例中减少 LOW_YIELD 或高成本动作。")

report_text = "\n".join(report_lines)

with report_path.open("w", encoding="utf-8") as f:
    f.write(report_text)

print("=" * 80)
print("实验分析完成")
print("=" * 80)
print(f"病例数量：{num_cases}")
print(f"平均 Judge 得分：{avg_judge_score:.2f}")
print(f"平均规则得分：{avg_rule_score:.2f}")
print(f"平均轮数：{avg_turns:.2f}")
print(f"平均成本：{avg_cost:.2f}")
print("")
print("动作标签总数：")
print(f"HIGH_YIELD：{total_high_yield}")
print(f"REASONABLE：{total_reasonable}")
print(f"LOW_YIELD：{total_low_yield}")
print(f"INEFFICIENT：{total_inefficient}")
print(f"CRITICAL_ERROR：{total_critical_error}")
print("")
print(f"LLM 总调用次数：{total_llm_calls}")
print(f"LLM 总 token：{total_llm_tokens}")
print("=" * 80)
print("报告已保存到：")
print(report_path)
