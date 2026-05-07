import json
from pathlib import Path
import sys

# 找到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 把 src 加入 Python 搜索路径
sys.path.append(str(PROJECT_ROOT / "src"))

from patient_agent import PatientAgent


case_path = PROJECT_ROOT / "data" / "cases.json"

with case_path.open("r", encoding="utf-8") as f:
    cases = json.load(f)

# 先测试第一个病例：社区获得性肺炎
case = cases[0]

patient = PatientAgent(case)

print("=" * 50)
print("医生一开始看到的信息：")
print(patient.get_initial_info())

print("=" * 50)
print("医生开始问诊：")

questions = [
    "患者年龄多大？",
    "患者有没有基础病史？",
    "有没有发热？",
    "有没有咳嗽？",
    "有没有咳痰？",
    "有没有胸痛？",
    "有没有呼吸困难？",
    "肺部听诊有什么异常？"
]

for question in questions:
    answer = patient.answer_question(question)
    print(f"医生：{question}")
    print(f"患者：{answer}")
    print("-" * 50)

