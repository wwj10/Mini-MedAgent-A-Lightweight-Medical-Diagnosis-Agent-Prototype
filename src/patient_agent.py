from typing import Dict, Any

class PatientAgent:
    """
    PatientAgent 模拟病人。
    它拿到完整隐藏病例，但只能根据医生的问题回答相关信息。
    """

    def __init__(self, case: Dict[str, Any]):
        self.case = case
        self.profile = case["hidden_profile"]

    def get_initial_info(self) -> str:
        """
        返回医生一开始能看到的信息：主诉。
        """
        return self.case["chief_complaint"]

    def answer_question(self, question: str) -> str:
        """
        根据医生的问题，从隐藏病例中返回相关信息。
        当前版本先用关键词匹配，后面再升级成 LLM Patient Agent。
        """
        question = question.lower()

        age = self.profile.get("age", "未知")
        sex = self.profile.get("sex", "未知")
        history = self.profile.get("history", "无特殊病史")
        symptoms = self.profile.get("symptoms", {})
        physical_exam = self.profile.get("physical_exam", {})

        # 基本信息
        if "年龄" in question or "多大" in question or "age" in question:
            return f"患者年龄：{age}岁。"

        if "性别" in question or "男" in question or "女" in question or "sex" in question:
            return f"患者性别：{sex}。"

        if "病史" in question or "既往" in question or "基础病" in question or "history" in question:
            return f"既往史：{history}。"

        # 症状相关
        if "发热" in question or "体温" in question or "fever" in question:
            return symptoms.get("fever", physical_exam.get("temperature", "未提及明显发热。"))

        if "咳嗽" in question or "咳" in question or "cough" in question:
            return symptoms.get("cough", "未提及明显咳嗽。")

        if "咳痰" in question or "痰" in question or "sputum" in question:
            return symptoms.get("sputum", "未提及明显咳痰。")

        if "胸痛" in question or "chest pain" in question:
            return symptoms.get("chest_pain", "未提及明显胸痛。")

        if "呼吸困难" in question or "气促" in question or "dyspnea" in question:
            return symptoms.get("dyspnea", "未提及明显呼吸困难。")

        if "出汗" in question or "大汗" in question or "sweating" in question:
            return symptoms.get("sweating", "未提及明显出汗。")

        if "恶心" in question or "nausea" in question:
            return symptoms.get("nausea", "未提及明显恶心。")

        if "呕吐" in question or "vomiting" in question:
            return symptoms.get("vomiting", "未提及明显呕吐。")

        if "腹痛" in question or "肚子痛" in question or "abdominal pain" in question:
            return symptoms.get("abdominal_pain", "未提及明显腹痛。")

        if "多饮" in question or "口渴" in question or "polydipsia" in question:
            return symptoms.get("polydipsia", "未提及明显多饮。")

        if "多尿" in question or "夜尿" in question or "polyuria" in question:
            return symptoms.get("polyuria", "未提及明显多尿。")

        if "体重" in question or "消瘦" in question or "weight" in question:
            return symptoms.get("weight_loss", "未提及明显体重变化。")

        if "咯血" in question or "hemoptysis" in question:
            return symptoms.get("hemoptysis", "未提及明显咯血。")

        if "腿" in question or "下肢" in question or "小腿" in question:
            return symptoms.get("leg_pain", physical_exam.get("leg_exam", "未提及明显下肢异常。"))

        # 查体相关
        if "血压" in question or "blood pressure" in question:
            return physical_exam.get("blood_pressure", "未记录血压异常。")

        if "心率" in question or "heart rate" in question:
            return physical_exam.get("heart_rate", "未记录心率异常。")

        if "肺部听诊" in question or "听诊" in question or "啰音" in question:
            return physical_exam.get("lung_auscultation", "未见明显肺部听诊异常。")

        if "麦氏点" in question or "右下腹压痛" in question:
            return physical_exam.get("mcburney_tenderness", "未记录麦氏点压痛。")

        if "反跳痛" in question:
            return physical_exam.get("rebound_tenderness", "未记录反跳痛。")

        if "血氧" in question or "spo2" in question:
            return physical_exam.get("spo2", "未记录血氧饱和度异常。")

        # 如果没有匹配到
        return "这个问题我暂时无法明确回答，病例中没有直接相关信息。"
