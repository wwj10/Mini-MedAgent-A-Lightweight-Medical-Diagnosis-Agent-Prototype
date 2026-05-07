import json
from memory_manager import MemoryManager
from judge_agent import JudgeAgent
from pathlib import Path
from typing import Dict, Any, List
from process_grader import ProcessGrader

from patient_agent import PatientAgent
from exam_agent import ExamAgent
from doctor_agent import DoctorAgent


class EpisodeRunner:
    """
    EpisodeRunner 负责运行一个完整病例。
    它把 DoctorAgent、PatientAgent、ExamAgent 串起来。
    """

    def __init__(
            self,
            case: Dict[str, Any],
            max_turns: int = 8,
            use_memory: bool = True,
            memory_mode: str = "all",
            memory_retrieval_mode: str = "keyword",
            memory_top_k: int = 5,
            memory_similarity_threshold: float = 0.35,
            memory_prompt_style: str = "full",
    ):

        self.case = case
        self.max_turns = max_turns

        self.patient_agent = PatientAgent(case)
        self.exam_agent = ExamAgent(case)
        self.doctor_agent = DoctorAgent()

        self.judge_agent = JudgeAgent()
        self.process_grader = ProcessGrader()

        self.use_memory = use_memory
        self.memory_mode = memory_mode
        self.memory_retrieval_mode = memory_retrieval_mode
        self.memory_top_k = memory_top_k
        self.memory_similarity_threshold = memory_similarity_threshold
        self.memory_prompt_style = memory_prompt_style

        project_root = Path(__file__).resolve().parents[1]
        memory_path = project_root / "outputs" / "memory" / "medical_memory.json"

        self.memory_manager = (
            MemoryManager(
                memory_path=memory_path,
                retrieval_mode=memory_retrieval_mode,
            )
            if use_memory
            else None
        )

        self.memory_added_count = 0

        self.judgement = None
        self.rule_score = None
        self.process_grading = None

        self.chief_complaint = case["chief_complaint"]
        self.gold_diagnosis = case["hidden_profile"]["diagnosis"]

        self.transcript: List[Dict[str, Any]] = []
        self.final_diagnosis = None

    def run(self) -> Dict[str, Any]:
        print("=" * 60)
        print("开始新病例")
        print("病例ID：", self.case["case_id"])
        print("患者主诉：", self.chief_complaint)
        print("=" * 60)

        for turn in range(1, self.max_turns + 1):
            print(f"\n第 {turn} 轮")

            if self.memory_manager is not None:
                memory_context = self.memory_manager.format_relevant_memories(
                    chief_complaint=self.chief_complaint,
                    transcript=self.transcript,
                    top_k=self.memory_top_k,
                    lesson_type_filter=self.memory_mode,
                    retrieval_mode=self.memory_retrieval_mode,
                    similarity_threshold=self.memory_similarity_threshold,
                    prompt_style=self.memory_prompt_style,
                )



            else:
                memory_context = "未启用经验记忆。"

            action = self.doctor_agent.decide_next_action(
                chief_complaint=self.chief_complaint,
                transcript=self.transcript,
                total_cost=self.exam_agent.get_total_cost(),
                memory_context=memory_context,
            )

            action_type = action["action_type"]
            content = action["content"]
            reason = action.get("reason", "")

            print("DoctorAgent 动作：", action_type)
            print("内容：", content)
            print("理由：", reason)

            if action_type == "AskQuestion":
                observation = self.patient_agent.answer_question(content)

                record = {
                    "turn": turn,
                    "action_type": action_type,
                    "content": content,
                    "reason": reason,
                    "observation": observation,
                }

                self.transcript.append(record)

                print("PatientAgent 回答：", observation)

            elif action_type == "OrderTest":
                test_result = self.exam_agent.order_test(content)

                observation = test_result["result"]

                record = {
                    "turn": turn,
                    "action_type": action_type,
                    "content": content,
                    "reason": reason,
                    "test_name": test_result["test_name"],
                    "observation": observation,
                    "cost": test_result["cost"],
                    "total_cost": test_result["total_cost"],
                    "success": test_result["success"],
                }

                self.transcript.append(record)

                print("ExamAgent 检查项目：", test_result["test_name"])
                print("ExamAgent 检查结果：", observation)
                print("本次成本：", test_result["cost"])
                print("累计成本：", test_result["total_cost"])

            elif action_type == "SubmitDiagnosis":
                self.final_diagnosis = content

                record = {
                    "turn": turn,
                    "action_type": action_type,
                    "content": content,
                    "reason": reason,
                    "observation": "DoctorAgent 已提交最终诊断。",
                }

                self.transcript.append(record)

                print("最终诊断：", self.final_diagnosis)
                break

            else:
                print("未知动作，终止病例。")
                break

        self.rule_score = self._simple_score(
            prediction=self.final_diagnosis,
            gold=self.gold_diagnosis,
        )

        self.judgement = self.judge_agent.evaluate(
            chief_complaint=self.chief_complaint,
            gold_diagnosis=self.gold_diagnosis,
            final_diagnosis=self.final_diagnosis,
            transcript=self.transcript,
        )

        self.process_grading = self.process_grader.grade_actions(
            chief_complaint=self.chief_complaint,
            gold_diagnosis=self.gold_diagnosis,
            final_diagnosis=self.final_diagnosis,
            judge_score=self.judgement["score"],
            total_cost=self.exam_agent.get_total_cost(),
            transcript=self.transcript,
        )

        result = self._build_result()

        if self.memory_manager is not None:
            self.memory_added_count = self.memory_manager.add_from_case_result(result)
            self.memory_manager.save()
            result = self._build_result()

        self._print_summary(result)

        return result

    def _build_result(self) -> Dict[str, Any]:
        if self.rule_score is None:
            self.rule_score = self._simple_score(
                prediction=self.final_diagnosis,
                gold=self.gold_diagnosis,
            )

        if self.judgement is None:
            self.judgement = {
                "score": self.rule_score,
                "label": "RULE_ONLY",
                "reason": "尚未调用 JudgeAgent，仅使用规则评分。",
            }

        if self.process_grading is None:
            self.process_grading = {
                "overall_comment": "尚未调用 ProcessGrader。",
                "action_reviews": [],
            }

        return {
            "case_id": self.case["case_id"],
            "chief_complaint": self.chief_complaint,
            "gold_diagnosis": self.gold_diagnosis,
            "final_diagnosis": self.final_diagnosis,
            "score": self.judgement["score"],
            "rule_score": self.rule_score,
            "judge_label": self.judgement["label"],
            "judge_reason": self.judgement["reason"],
            "total_cost": self.exam_agent.get_total_cost(),
            "ordered_tests": self.exam_agent.get_ordered_tests(),
            "turns": len(self.transcript),
            "llm_usage": self.doctor_agent.get_usage(),
            "judge_llm_usage": self.judge_agent.get_usage(),
            "process_overall_comment": self.process_grading["overall_comment"],
            "action_reviews": self.process_grading["action_reviews"],
            "process_grader_usage": self.process_grader.get_usage(),
            "use_memory": self.use_memory,
            "memory_mode": self.memory_mode,
            "memory_retrieval_mode": self.memory_retrieval_mode,
            "memory_top_k": self.memory_top_k,
            "memory_similarity_threshold": self.memory_similarity_threshold,
            "memory_prompt_style": self.memory_prompt_style,
            "memory_added_count": self.memory_added_count,
            "memory_total_count": self.memory_manager.count() if self.memory_manager is not None else 0,
            "transcript": self.transcript,
        }

    def _simple_score(self, prediction: str | None, gold: str) -> int:
        """
        暂时用简单规则评分。
        后面我们会升级成 JudgeAgent。
        """
        if prediction is None:
            return 0

        pred = prediction.strip()
        gold = gold.strip()

        if pred == gold:
            return 100

        if gold in pred or pred in gold:
            return 80

        return 0

    def _print_summary(self, result: Dict[str, Any]) -> None:
        usage = result["llm_usage"]
        judge_usage = result["judge_llm_usage"]
        process_usage = result["process_grader_usage"]

        label_counts = {}

        for review in result["action_reviews"]:
            label = review.get("label", "UNKNOWN")
            label_counts[label] = label_counts.get(label, 0) + 1

        print("\n" + "=" * 60)
        print("病例结束")
        print("病例ID：", result["case_id"])
        print("标准诊断：", result["gold_diagnosis"])
        print("Agent诊断：", result["final_diagnosis"])
        print("规则得分：", result["rule_score"])
        print("Judge得分：", result["score"])
        print("Judge标签：", result["judge_label"])
        print("Judge理由：", result["judge_reason"])
        print("ProcessGrader总评：", result["process_overall_comment"])
        print("动作标签统计：", label_counts)
        print("总轮数：", result["turns"])
        print("总成本：", result["total_cost"])
        print("申请过的检查：", result["ordered_tests"])
        print("Doctor LLM调用次数：", usage["call_count"])
        print("Doctor LLM总token：", usage["total_tokens"])
        print("Judge LLM调用次数：", judge_usage["call_count"])
        print("Judge LLM总token：", judge_usage["total_tokens"])
        print("ProcessGrader LLM调用次数：", process_usage["call_count"])
        print("ProcessGrader LLM总token：", process_usage["total_tokens"])
        print("是否启用记忆：", result["use_memory"])
        print("记忆模式：", result["memory_mode"])
        print("记忆检索方式：", result["memory_retrieval_mode"])
        print("记忆 top_k：", result["memory_top_k"])
        print("记忆相似度阈值：", result["memory_similarity_threshold"])
        print("记忆注入格式：", result["memory_prompt_style"])
        print("本病例新增经验数：", result["memory_added_count"])
        print("当前经验总数：", result["memory_total_count"])
        print("=" * 60)

    def save_transcript(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self._build_result()
        output_path = output_dir / f"{self.case['case_id']}_transcript.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return output_path
