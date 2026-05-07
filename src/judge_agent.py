import json
from typing import Dict, Any, List

from llm_client import LLMClient


class JudgeAgent:
    """
    JudgeAgent 用于评估 DoctorAgent 的最终诊断是否正确。
    它可以看到标准诊断，但它只用于评估，不参与诊断过程。
    """

    def __init__(self):
        self.llm = LLMClient()

    def evaluate(
        self,
        chief_complaint: str,
        gold_diagnosis: str,
        final_diagnosis: str | None,
        transcript: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if final_diagnosis is None:
            return {
                "score": 0,
                "label": "NO_DIAGNOSIS",
                "reason": "DoctorAgent 未提交最终诊断。",
            }

        transcript_text = self._format_transcript(transcript)

        system_prompt = """
你是一个医学诊断评估器 JudgeAgent。

你的任务是比较：
1. 标准诊断 gold_diagnosis
2. Agent 提交的 final_diagnosis

并判断 Agent 的诊断在医学语义上是否正确。

评分标准：
- 100：完全正确，或语义等价，或 Agent 诊断是标准诊断的合理更具体表述。
  例如：标准诊断是“急性心肌梗死”，Agent 诊断是“急性下壁ST段抬高型心肌梗死”，应接近或等于100。
  例如：标准诊断是“肺栓塞”，Agent 诊断是“急性肺血栓栓塞症”，应接近或等于100。
- 80：基本正确，但表述略泛化或缺少关键分型。
  例如：标准诊断是“2型糖尿病”，Agent 诊断是“糖尿病”。
- 50：方向相关，但诊断不够准确，可能是相近鉴别诊断。
- 0：明显错误，或与标准诊断无关。

请只返回严格 JSON，不要输出 Markdown，不要输出额外解释。
"""

        user_prompt = f"""
患者主诉：
{chief_complaint}

标准诊断：
{gold_diagnosis}

Agent最终诊断：
{final_diagnosis}

诊断过程记录：
{transcript_text}

请给出评分。
返回格式如下：
{{
  "score": 0到100之间的整数,
  "label": "CORRECT / PARTIALLY_CORRECT / RELATED / WRONG / NO_DIAGNOSIS",
  "reason": "一句话说明评分理由"
}}
"""

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

        raw_response = self.llm.chat(messages=messages, temperature=0.0)
        return self._parse_judgement(raw_response)

    def _format_transcript(self, transcript: List[Dict[str, Any]]) -> str:
        if not transcript:
            return "暂无诊断过程记录。"

        lines = []

        for item in transcript:
            turn = item.get("turn", "")
            action_type = item.get("action_type", "")
            content = item.get("content", "")
            observation = item.get("observation", "")
            cost = item.get("cost", None)

            if cost is None:
                lines.append(
                    f"第{turn}轮：动作={action_type}，内容={content}，观察结果={observation}"
                )
            else:
                lines.append(
                    f"第{turn}轮：动作={action_type}，内容={content}，观察结果={observation}，成本={cost}"
                )

        return "\n".join(lines)

    def _parse_judgement(self, raw_response: str) -> Dict[str, Any]:
        try:
            text = raw_response.strip()

            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()

            start = text.find("{")
            end = text.rfind("}")

            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1]

            judgement = json.loads(text)

            score = int(judgement.get("score", 0))
            score = max(0, min(100, score))

            label = judgement.get("label", "UNKNOWN")
            reason = judgement.get("reason", "")

            return {
                "score": score,
                "label": label,
                "reason": reason,
            }

        except Exception as e:
            return {
                "score": 0,
                "label": "PARSE_ERROR",
                "reason": f"JudgeAgent 输出解析失败：{e}。原始输出：{raw_response}",
            }

    def get_usage(self) -> Dict[str, int | str]:
        return self.llm.get_usage()
