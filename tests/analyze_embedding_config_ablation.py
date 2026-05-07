from pathlib import Path
import statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
report_path = PROJECT_ROOT / "outputs" / "embedding_config_ablation" / "embedding_config_repeated_report.md"

results = {
    "top5_threshold035": {
        "score": [96.00, 100.00, 100.00],
        "turns": [4.60, 5.40, 5.40],
        "cost": [8.60, 10.80, 9.80],
        "low_yield": [0, 1, 0],
        "doctor_tokens": [33382, 38156, 38994],
    },
    "top3_threshold040": {
        "score": [100.00, 96.00, 96.00],
        "turns": [5.40, 5.20, 4.60],
        "cost": [9.20, 8.80, 7.60],
        "low_yield": [2, 1, 1],
        "doctor_tokens": [33907, 33405, 28598],
    },
    "top2_threshold045": {
        "score": [96.00, 100.00, 96.00],
        "turns": [5.20, 4.80, 5.20],
        "cost": [8.20, 8.60, 10.00],
        "low_yield": [1, 1, 1],
        "doctor_tokens": [28520, 25987, 29208],
    },
}


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


rows = []

for config_name, metrics in results.items():
    row = {
        "config": config_name,
        "score_mean": mean(metrics["score"]),
        "score_std": std(metrics["score"]),
        "turns_mean": mean(metrics["turns"]),
        "turns_std": std(metrics["turns"]),
        "cost_mean": mean(metrics["cost"]),
        "cost_std": std(metrics["cost"]),
        "low_yield_mean": mean(metrics["low_yield"]),
        "low_yield_std": std(metrics["low_yield"]),
        "tokens_mean": mean(metrics["doctor_tokens"]),
        "tokens_std": std(metrics["doctor_tokens"]),
    }
    rows.append(row)

best_score = max(rows, key=lambda x: x["score_mean"])
best_turns = min(rows, key=lambda x: x["turns_mean"])
best_cost = min(rows, key=lambda x: x["cost_mean"])
best_low_yield = min(rows, key=lambda x: x["low_yield_mean"])
best_tokens = min(rows, key=lambda x: x["tokens_mean"])

lines = []

lines.append("# Embedding Memory 配置重复实验稳定性报告")
lines.append("")
lines.append("## 1. 实验目的")
lines.append("")
lines.append("由于 LLM 输出存在随机性，单次实验结果可能不稳定。本报告汇总 3 次重复运行结果，用于判断不同 embedding memory 配置的稳定表现。")
lines.append("")
lines.append("比较配置包括：")
lines.append("")
lines.append("- top5_threshold035：top_k=5，similarity_threshold=0.35")
lines.append("- top3_threshold040：top_k=3，similarity_threshold=0.40")
lines.append("- top2_threshold045：top_k=2，similarity_threshold=0.45")
lines.append("")

lines.append("## 2. 三次重复实验均值与标准差")
lines.append("")
lines.append("| 配置 | 平均得分 | 得分SD | 平均轮数 | 轮数SD | 平均成本 | 成本SD | 平均LOW_YIELD | LOW_YIELD SD | 平均Doctor tokens | Tokens SD |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for row in rows:
    lines.append(
        "| {config} | {score_mean:.2f} | {score_std:.2f} | {turns_mean:.2f} | {turns_std:.2f} | {cost_mean:.2f} | {cost_std:.2f} | {low_yield_mean:.2f} | {low_yield_std:.2f} | {tokens_mean:.0f} | {tokens_std:.0f} |".format(
            **row
        )
    )

lines.append("")

lines.append("## 3. 最优指标")
lines.append("")
lines.append(f"- 最高平均得分：{best_score['config']}，平均得分 {best_score['score_mean']:.2f}")
lines.append(f"- 最低平均轮数：{best_turns['config']}，平均轮数 {best_turns['turns_mean']:.2f}")
lines.append(f"- 最低平均成本：{best_cost['config']}，平均成本 {best_cost['cost_mean']:.2f}")
lines.append(f"- 最低 LOW_YIELD：{best_low_yield['config']}，平均 LOW_YIELD {best_low_yield['low_yield_mean']:.2f}")
lines.append(f"- 最低 Doctor tokens：{best_tokens['config']}，平均 tokens {best_tokens['tokens_mean']:.0f}")
lines.append("")

lines.append("## 4. 结果解释")
lines.append("")
lines.append("三次重复实验表明，单次实验中的最优配置并不一定稳定。")
lines.append("")
lines.append("top5_threshold035 在平均得分和 LOW_YIELD 控制方面表现最好，说明更多经验注入有助于提升诊断质量和减少低价值动作。但它的平均成本和 token 消耗较高。")
lines.append("")
lines.append("top3_threshold040 在单次实验中曾取得最高得分，但重复实验后平均得分下降，LOW_YIELD 数量也更高，说明该配置稳定性不如单次结果显示得那么强。")
lines.append("")
lines.append("top2_threshold045 的 token 消耗最低，说明更严格的相似度阈值和更少 memory 注入可以降低上下文成本，但它没有取得最高平均得分。")
lines.append("")

lines.append("## 5. 推荐默认配置")
lines.append("")
lines.append("综合医学安全性、诊断准确率和动作级质量，当前推荐默认配置为：")
lines.append("")
lines.append("```python")
lines.append("memory_top_k = 5")
lines.append("memory_similarity_threshold = 0.35")
lines.append("```")
lines.append("")
lines.append("理由是：医学诊断 Agent 应优先保证诊断准确性和过程质量。top5_threshold035 在三次重复实验中取得最高平均得分和最低 LOW_YIELD。")
lines.append("")
lines.append("不过，如果后续目标是降低 API 成本，可以考虑 top2_threshold045，因为它的 Doctor tokens 最低。")
lines.append("")

lines.append("## 6. 下一步")
lines.append("")
lines.append("下一步可以优化 memory 文本压缩，让 top5_threshold035 保持诊断质量的同时降低 token 消耗。具体做法是将注入 prompt 的 memory 从完整经验压缩为一句短 lesson。")

report_text = "\n".join(lines)

report_path.parent.mkdir(parents=True, exist_ok=True)

with report_path.open("w", encoding="utf-8") as f:
    f.write(report_text)

print("=" * 100)
print("重复实验稳定性报告生成成功")
print("=" * 100)
print("报告路径：")
print(report_path)
print("")
print("推荐默认配置：")
print("memory_top_k = 5")
print("memory_similarity_threshold = 0.35")
