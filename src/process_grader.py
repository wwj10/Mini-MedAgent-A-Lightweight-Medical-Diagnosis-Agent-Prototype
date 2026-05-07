import json
from typing import Dict, Any, List

from llm_client import LLMClient


class ProcessGrader:
    """
    ProcessGrader 对一次完整诊断过程中的每一步动作进行评价。
    它用于事后分析，不参与 DoctorAgent 的实时诊断。
    """

    def __init__(self):
        self.llm = LLMClient()

    def grade_actions(
        self,
        chief_complaint: str,
        gold_diagnosis: str,
        final_diagnosis: str | None,
        judge_score: int,
        total_cost: int,
        transcript: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        transcript_text = self._format_transcript(transcript)

        system_prompt = """
你是一个医学诊断过程评估器 ProcessGrader。

你的任务不是重新诊断，而是评价 DoctorAgent 在诊断过程中的每一步动作是否合理。

你需要根据：
1. 患者主诉
2. 标准诊断
3. Agent 最终诊断
4. Judge 最终得分
5. 检查总成本
6. 完整诊断轨迹 transcript

对每一步动作打标签。

可选标签：

1. HIGH_YIELD
表示该动作对诊断非常有帮助。
例如：胸痛病例中及时申请心电图；肺炎病例中申请胸片；肺栓塞病例中申请D-二聚体或CTPA。

2. REASONABLE
表示该动作合理，但不是最关键动作。
例如：询问基础病史、伴随症状等。

3. LOW_YIELD
表示该动作信息价值有限，但不算严重错误。
例如：重复询问已知信息，或询问与当前诊断关系不大的症状。

4. INEFFICIENT
表示该动作成本较高、顺序不佳、或可能存在过度检查。
例如：没有先做基础检查就直接做昂贵检查。

5. CRITICAL_ERROR
表示该动作明显错误，可能误导诊断或延误关键处理。
例如：胸痛伴ST段抬高却忽视心梗方向。

请注意：
- 不要因为最终诊断正确，就把所有动作都评为 HIGH_YIELD。
- 不要因为最终诊断错误，就把所有动作都评为 CRITICAL_ERROR。
- 要逐步分析每个动作本身的价值。
- 检查动作需要考虑收益和成本。
- 提交诊断动作也需要评价：如果证据充分且诊断正确，是 HIGH_YIELD；如果过早提交，是 INEFFICIENT 或 CRITICAL_ERROR。
- 必须返回严格 JSON，不要输出 Markdown，不要输出额外解释。
"""

        user_prompt = f"""
患者主诉：
{chief_complaint}

标准诊断：
{gold_diagnosis}

Agent最终诊断：
{final_diagnosis}

Judge最终得分：
{judge_score}

检查总成本：
{total_cost}

完整诊断轨迹：
{transcript_text}

请对每一步动作进行评价。

返回格式如下：
{{
  "overall_comment": "一句话总结整个诊断过程",
  "action_reviews": [
    {{
      "turn": 1,
      "action_type": "AskQuestion / OrderTest / SubmitDiagnosis",
      "content": "该轮动作内容",
      "label": "HIGH_YIELD / REASONABLE / LOW_YIELD / INEFFICIENT / CRITICAL_ERROR",
      "reason": "为什么给这个标签",
      "suggestion": "如果要改进，这一步应该怎么做；如果无需改进，写无需改进"
    }}
  ]
}}
"""

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

        raw_response = self.llm.chat(messages=messages, temperature=0.0)
        return self._parse_grading(raw_response, transcript)

    def _format_transcript(self, transcript: List[Dict[str, Any]]) -> str:
        if not transcript:
            return "暂无诊断轨迹。"

        lines = []

        for item in transcript:
            turn = item.get("turn", "")
            action_type = item.get("action_type", "")
            content = item.get("content", "")
            reason = item.get("reason", "")
            observation = item.get("observation", "")
            cost = item.get("cost", None)
            total_cost = item.get("total_cost", None)

            if cost is None:
                lines.append(
                    f"第{turn}轮：动作={action_type}，内容={content}，理由={reason}，观察结果={observation}"
                )
            else:
                lines.append(
                    f"第{turn}轮：动作={action_type}，内容={content}，理由={reason}，观察结果={observation}，本次成本={cost}，累计成本={total_cost}"
                )

        return "\n".join(lines)

    def _parse_grading(
        self,
        raw_response: str,
        transcript: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            text = raw_response.strip()

            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()

            start = text.find("{")
            end = text.rfind("}")

            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1]

            data = json.loads(text)

            overall_comment = data.get("overall_comment", "")
            action_reviews = data.get("action_reviews", [])

            valid_labels = {
                "HIGH_YIELD",
                "REASONABLE",
                "LOW_YIELD",
                "INEFFICIENT",
                "CRITICAL_ERROR",
            }

            cleaned_reviews = []

            for idx, item in enumerate(action_reviews):
                label = item.get("label", "REASONABLE")

                if label not in valid_labels:
                    label = "REASONABLE"

                cleaned_reviews.append({
                    "turn": item.get("turn", idx + 1),
                    "action_type": item.get("action_type", ""),
                    "content": item.get("content", ""),
                    "label": label,
                    "reason": item.get("reason", ""),
                    "suggestion": item.get("suggestion", ""),
                })

            return {
                "overall_comment": overall_comment,
                "action_reviews": cleaned_reviews,
            }

        except Exception as e:
            fallback_reviews = []

            for item in transcript:
                fallback_reviews.append({
                    "turn": item.get("turn", ""),
                    "action_type": item.get("action_type", ""),
                    "content": item.get("content", ""),
                    "label": "REASONABLE",
                    "reason": f"ProcessGrader 输出解析失败，使用默认标签。错误：{e}",
                    "suggestion": "需要人工复核。",
                })

            return {
                "overall_comment": "ProcessGrader 输出解析失败，已使用默认动作评价。",
                "action_reviews": fallback_reviews,
            }

    def get_usage(self) -> Dict[str, int | str]:
        return self.llm.get_usage()
