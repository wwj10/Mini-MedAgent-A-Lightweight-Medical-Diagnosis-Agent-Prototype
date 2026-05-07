import json
from pathlib import Path

# 当前文件位置：D:\med-agent-learning\tests\check_cases.py
# parents[1] 就是项目根目录：D:\med-agent-learning
PROJECT_ROOT = Path(__file__).resolve().parents[1]

case_path = PROJECT_ROOT / "data" / "cases.json"

print("当前读取路径：", case_path)

if not case_path.exists():
    raise FileNotFoundError(f"没有找到病例文件：{case_path}")

with case_path.open("r", encoding="utf-8") as f:
    cases = json.load(f)

print(f"成功读取 {len(cases)} 个病例。")

for case in cases:
    print("-" * 40)
    print("病例ID:", case["case_id"])
    print("主诉:", case["chief_complaint"])
    print("标准诊断:", case["hidden_profile"]["diagnosis"])
