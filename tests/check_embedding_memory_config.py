import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from episode_runner import EpisodeRunner


case_path = PROJECT_ROOT / "data" / "cases.json"

with case_path.open("r", encoding="utf-8") as f:
    cases = json.load(f)

case = cases[0]

configs = [
    {
        "name": "top5_threshold035",
        "top_k": 5,
        "threshold": 0.35,
    },
    {
        "name": "top3_threshold040",
        "top_k": 3,
        "threshold": 0.40,
    },
    {
        "name": "top2_threshold045",
        "top_k": 2,
        "threshold": 0.45,
    },
]

for config in configs:
    print("\n" + "=" * 100)
    print("开始测试配置：", config["name"])
    print("top_k =", config["top_k"])
    print("threshold =", config["threshold"])
    print("=" * 100)

    runner = EpisodeRunner(
        case=case,
        max_turns=8,
        use_memory=True,
        memory_mode="all",
        memory_retrieval_mode="embedding",
        memory_top_k=config["top_k"],
        memory_similarity_threshold=config["threshold"],
    )

    result = runner.run()

    output_dir = PROJECT_ROOT / "outputs" / "embedding_config_test" / config["name"]
    output_path = runner.save_transcript(output_dir)

    print("\n配置测试完成：", config["name"])
    print("诊断：", result["final_diagnosis"])
    print("得分：", result["score"])
    print("轮数：", result["turns"])
    print("成本：", result["total_cost"])
    print("Doctor tokens：", result["llm_usage"]["total_tokens"])
    print("结果保存到：", output_path)
