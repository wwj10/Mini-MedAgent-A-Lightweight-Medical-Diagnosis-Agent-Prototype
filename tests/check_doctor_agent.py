import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from doctor_agent import DoctorAgent


case_path = PROJECT_ROOT / "data" / "cases.json"

with case_path.open("r", encoding="utf-8") as f:
    cases = json.load(f)

# 先测试第一个病例：发热、咳嗽3天
case = cases[0]
chief_complaint = case["chief_complaint"]

doctor = DoctorAgent()

print("=" * 50)
print("患者主诉：", chief_complaint)
print("=" * 50)

# 场景1：没有任何历史信息，看看 DoctorAgent 第一步想做什么
transcript = []
action = doctor.decide_next_action(
    chief_complaint=chief_complaint,
    transcript=transcript,
    total_cost=0,
)

print("场景1：暂无历史交互")
print("DoctorAgent 决策：")
print(action)

print("=" * 50)

# 场景2：已经问过一些问题
transcript = [
    {
        "action_type": "AskQuestion",
        "content": "有没有发热？",
        "observation": "体温最高38.5℃。",
    },
    {
        "action_type": "AskQuestion",
        "content": "有没有咳痰？",
        "observation": "少量黄痰。",
    },
    {
        "action_type": "AskQuestion",
        "content": "有没有呼吸困难？",
        "observation": "无明显呼吸困难。",
    },
]

action = doctor.decide_next_action(
    chief_complaint=chief_complaint,
    transcript=transcript,
    total_cost=0,
)

print("场景2：已有问诊信息")
print("DoctorAgent 决策：")
print(action)

print("=" * 50)

# 场景3：已经有问诊 + 胸片结果
transcript = [
    {
        "action_type": "AskQuestion",
        "content": "有没有发热？",
        "observation": "体温最高38.5℃。",
    },
    {
        "action_type": "AskQuestion",
        "content": "有没有咳痰？",
        "observation": "少量黄痰。",
    },
    {
        "action_type": "OrderTest",
        "content": "胸片",
        "observation": "右下肺片状阴影",
        "cost": 5,
    },
]

action = doctor.decide_next_action(
    chief_complaint=chief_complaint,
    transcript=transcript,
    total_cost=5,
)

print("场景3：已有问诊和胸片结果")
print("DoctorAgent 决策：")
print(action)
