import json
from typing import List, Dict, Any

from llm_client import LLMClient


class DoctorAgent:
    """
    DoctorAgent 根据当前病例信息和历史交互，决定下一步行动：
    1. AskQuestion：问诊
    2. OrderTest：申请检查
    3. SubmitDiagnosis：提交诊断
    """

    def __init__(self):
        self.llm = LLMClient()

    def decide_next_action(
            self,
            chief_complaint: str,
            transcript: List[Dict[str, Any]],
            total_cost: int = 0,
            memory_context: str = "",
    ) -> Dict[str, str]:

        """
        根据当前已知信息，决定下一步动作。
        """

        transcript_text = self._format_transcript(transcript)

        system_prompt = """
你是一个医学诊断训练环境中的 DoctorAgent。

你不能直接看到完整病例，只能看到：
1. 患者主诉
2. 已经问过的问题和患者回答
3. 已经申请过的检查和检查结果
4. 当前累计检查成本

你的任务是决定下一步行动。

你只能选择以下三种动作之一：

1. AskQuestion
用于向患者询问病史、症状、体征相关信息。
例如：
{"action_type": "AskQuestion", "content": "有没有咳痰？", "reason": "发热咳嗽病例需要进一步判断感染可能。"}

2. OrderTest
用于申请检查。
可申请的检查包括：
血常规、CRP、胸片、胸部CT、心电图、肌钙蛋白、CK-MB、空腹血糖、随机血糖、糖化血红蛋白、尿糖、腹部超声、腹部CT、D-二聚体、血气分析、CTPA、下肢静脉超声。

例如：
{"action_type": "OrderTest", "content": "胸片", "reason": "发热咳嗽伴肺部症状，需要判断是否有肺部感染影像学证据。"}

3. SubmitDiagnosis
用于提交最终诊断。
例如：
{"action_type": "SubmitDiagnosis", "content": "社区获得性肺炎", "reason": "患者发热咳嗽，胸片有肺部浸润影，支持该诊断。"}

决策原则：
- 不要一开始就直接诊断，除非证据已经非常充分。
- 优先问高价值问题。
- 优先选择低成本、高收益检查。
- 不要滥用昂贵检查。
- 如果已有证据足以判断，应及时提交诊断。
- 每次只返回一个动作。
- 必须返回严格 JSON，不要输出 Markdown，不要输出解释文字。
- 可以参考历史经验记忆，但不要盲目照搬。
- 如果历史经验提示某些动作 LOW_YIELD 或 INEFFICIENT，应尽量避免重复类似低价值动作。
"""

        user_prompt = f"""
患者主诉：
{chief_complaint}

当前累计检查成本：
{total_cost}

可参考的历史经验记忆：
{memory_context if memory_context else "暂无相关历史经验。"}

历史交互记录：
{transcript_text}

请决定下一步行动。
只返回严格 JSON，格式如下：
{{
  "action_type": "AskQuestion 或 OrderTest 或 SubmitDiagnosis",
  "content": "你的具体问题、检查项目或诊断名称",
  "reason": "简短说明为什么这样做"
}}
"""

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

        raw_response = self.llm.chat(messages=messages, temperature=0.2)
        action = self._parse_action(raw_response)

        return action

    def get_usage(self) -> Dict[str, int | str]:
        return self.llm.get_usage()

    def _format_transcript(self, transcript: List[Dict[str, Any]]) -> str:
        if not transcript:
            return "暂无历史交互。"

        lines = []

        for idx, item in enumerate(transcript, start=1):
            action_type = item.get("action_type", "")
            content = item.get("content", "")
            observation = item.get("observation", "")
            cost = item.get("cost", None)

            if cost is None:
                lines.append(
                    f"第{idx}轮：动作={action_type}，内容={content}，观察结果={observation}"
                )
            else:
                lines.append(
                    f"第{idx}轮：动作={action_type}，内容={content}，观察结果={observation}，成本={cost}"
                )

        return "\n".join(lines)

    def _parse_action(self, raw_response: str) -> Dict[str, str]:
        """
        尽量从模型输出中解析 JSON。
        如果模型输出不规范，返回一个安全的默认动作。
        """

        try:
            text = raw_response.strip()

            # 防止模型偶尔包一层 ```json
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()

            # 提取第一个 { 到最后一个 }
            start = text.find("{")
            end = text.rfind("}")

            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1]

            action = json.loads(text)

            action_type = action.get("action_type", "")
            content = action.get("content", "")
            reason = action.get("reason", "")

            if action_type not in ["AskQuestion", "OrderTest", "SubmitDiagnosis"]:
                raise ValueError(f"未知 action_type: {action_type}")

            if not content:
                raise ValueError("content 为空。")

            return {
                "action_type": action_type,
                "content": content,
                "reason": reason,
            }

        except Exception as e:
            return {
                "action_type": "AskQuestion",
                "content": "请问还有哪些不舒服的症状？",
                "reason": f"模型输出解析失败，使用安全默认问诊动作。错误：{e}",
            }
