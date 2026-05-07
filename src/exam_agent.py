from typing import Dict, Any


class ExamAgent:
    """
    ExamAgent 模拟检查系统。
    医生输入检查项目，ExamAgent 根据隐藏病例返回对应检查结果，并记录成本。
    """

    def __init__(self, case: Dict[str, Any]):
        self.case = case
        self.profile = case["hidden_profile"]
        self.tests = self.profile.get("tests", {})
        self.total_cost = 0
        self.ordered_tests = []

        # 简单成本模型：数值越大代表检查越贵/越耗资源
        self.test_costs = {
            "blood_test": 3,
            "crp": 2,
            "chest_xray": 5,
            "chest_ct": 12,
            "ecg": 3,
            "troponin": 5,
            "ck_mb": 4,
            "fasting_glucose": 2,
            "random_glucose": 2,
            "hba1c": 4,
            "urine_glucose": 1,
            "abdominal_ultrasound": 6,
            "abdominal_ct": 12,
            "d_dimer": 4,
            "arterial_blood_gas": 5,
            "ctpa": 15,
            "lower_limb_ultrasound": 6,
        }

    def normalize_test_name(self, test_request: str) -> str | None:
        """
        把医生输入的自然语言检查名称，映射成病例 JSON 里的标准检查名。
        """
        request = test_request.lower()

        if "血常规" in request or "白细胞" in request or "blood" in request:
            return "blood_test"

        if "crp" in request or "c反应蛋白" in request:
            return "crp"

        if "胸片" in request or "胸部x线" in request or "xray" in request or "x-ray" in request:
            return "chest_xray"

        if "胸部ct" in request or "胸ct" in request or "chest ct" in request:
            return "chest_ct"

        if "心电图" in request or "ecg" in request:
            return "ecg"

        if "肌钙蛋白" in request or "troponin" in request:
            return "troponin"

        if "ck-mb" in request or "ck_mb" in request or "肌酸激酶" in request:
            return "ck_mb"

        if "空腹血糖" in request or "fasting glucose" in request:
            return "fasting_glucose"

        if "随机血糖" in request or "random glucose" in request:
            return "random_glucose"

        if "糖化血红蛋白" in request or "hba1c" in request:
            return "hba1c"

        if "尿糖" in request or "urine glucose" in request:
            return "urine_glucose"

        if "腹部超声" in request or "阑尾超声" in request or "abdominal ultrasound" in request:
            return "abdominal_ultrasound"

        if "腹部ct" in request or "abdominal ct" in request:
            return "abdominal_ct"

        if "d-二聚体" in request or "d二聚体" in request or "d_dimer" in request or "d-dimer" in request:
            return "d_dimer"

        if "血气" in request or "arterial blood gas" in request:
            return "arterial_blood_gas"

        if "ctpa" in request or "肺动脉cta" in request or "肺动脉ct" in request:
            return "ctpa"

        if "下肢超声" in request or "下肢静脉" in request or "lower limb ultrasound" in request:
            return "lower_limb_ultrasound"

        return None

    def order_test(self, test_request: str) -> Dict[str, Any]:
        """
        医生申请检查，返回结构化结果。
        """
        test_name = self.normalize_test_name(test_request)

        if test_name is None:
            return {
                "success": False,
                "test_name": None,
                "result": f"无法识别检查项目：{test_request}",
                "cost": 0,
                "total_cost": self.total_cost,
            }

        cost = self.test_costs.get(test_name, 5)

        if test_name not in self.tests:
            self.total_cost += cost
            self.ordered_tests.append(test_name)

            return {
                "success": False,
                "test_name": test_name,
                "result": f"该病例中没有提供 {test_name} 的检查结果。",
                "cost": cost,
                "total_cost": self.total_cost,
            }

        result = self.tests[test_name]

        self.total_cost += cost
        self.ordered_tests.append(test_name)

        return {
            "success": True,
            "test_name": test_name,
            "result": result,
            "cost": cost,
            "total_cost": self.total_cost,
        }

    def get_total_cost(self) -> int:
        return self.total_cost

    def get_ordered_tests(self) -> list:
        return self.ordered_tests
