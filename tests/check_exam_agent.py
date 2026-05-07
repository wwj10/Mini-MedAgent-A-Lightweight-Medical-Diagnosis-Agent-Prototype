import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from exam_agent import ExamAgent


case_path = PROJECT_ROOT / "data" / "cases.json"

with case_path.open("r", encoding="utf-8") as f:
    cases = json.load(f)

# 测试第一个病例：社区获得性肺炎
case = cases[0]
exam_agent = ExamAgent(case)

print("=" * 50)
print("病例主诉：", case["chief_complaint"])
print("开始申请检查")
print("=" * 50)

test_requests = [
    "申请血常规",
    "申请CRP",
    "申请胸片",
    "申请胸部CT",
    "申请心电图"
]

for request in test_requests:
    result = exam_agent.order_test(request)

    print(f"医生：{request}")
    print(f"检查项目：{result['test_name']}")
    print(f"检查结果：{result['result']}")
    print(f"本次成本：{result['cost']}")
    print(f"累计成本：{result['total_cost']}")
    print("-" * 50)

print("已经申请过的检查：", exam_agent.get_ordered_tests())
print("总成本：", exam_agent.get_total_cost())
