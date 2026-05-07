import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from judge_agent import JudgeAgent


judge = JudgeAgent()

test_cases = [
    {
        "chief_complaint": "突发胸痛2小时",
        "gold_diagnosis": "急性心肌梗死",
        "final_diagnosis": "急性下壁ST段抬高型心肌梗死",
    },
    {
        "chief_complaint": "突发呼吸困难伴胸痛1小时",
        "gold_diagnosis": "肺栓塞",
        "final_diagnosis": "急性肺血栓栓塞症",
    },
    {
        "chief_complaint": "多饮、多尿、体重下降1个月",
        "gold_diagnosis": "2型糖尿病",
        "final_diagnosis": "糖尿病",
    },
    {
        "chief_complaint": "发热、咳嗽3天",
        "gold_diagnosis": "社区获得性肺炎",
        "final_diagnosis": "急性心肌梗死",
    },
]

for item in test_cases:
    result = judge.evaluate(
        chief_complaint=item["chief_complaint"],
        gold_diagnosis=item["gold_diagnosis"],
        final_diagnosis=item["final_diagnosis"],
        transcript=[],
    )

    print("=" * 60)
    print("主诉：", item["chief_complaint"])
    print("标准诊断：", item["gold_diagnosis"])
    print("Agent诊断：", item["final_diagnosis"])
    print("Judge评分：", result["score"])
    print("标签：", result["label"])
    print("理由：", result["reason"])

print("=" * 60)
print("JudgeAgent LLM usage:")
print(judge.get_usage())
