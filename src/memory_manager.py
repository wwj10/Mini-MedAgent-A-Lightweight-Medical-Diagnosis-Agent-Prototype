import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class MemoryManager:
    """
    MemoryManager 负责把 ProcessGrader 的动作级反馈转化为可复用经验。
    当前版本是轻量规则版，不额外调用 LLM。
    """

    def __init__(
            self,
            memory_path: Path,
            retrieval_mode: str = "keyword",
            embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
    ):
        self.memory_path = memory_path
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        # retrieval_mode:
        # - keyword: 关键词检索
        # - tfidf: 本地 TF-IDF 向量检索
        # - embedding: 语义 embedding 检索
        self.retrieval_mode = retrieval_mode
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None

        # embedding 缓存：避免同一个 Episode 中反复编码 memory
        self._memory_embedding_cache_signature = None
        self._memory_embedding_cache_vectors = None

        self.memories: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if not self.memory_path.exists():
            self.memories = []
            return

        with self.memory_path.open("r", encoding="utf-8") as f:
            try:
                self.memories = json.load(f)
            except json.JSONDecodeError:
                self.memories = []

    def save(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        with self.memory_path.open("w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def add_from_case_result(self, case_result: Dict[str, Any]) -> int:
        """
        从一个病例结果中抽取经验。
        HIGH_YIELD 记为正向经验；
        LOW_YIELD / INEFFICIENT / CRITICAL_ERROR 记为负向经验。
        """
        action_reviews = case_result.get("action_reviews", [])
        added_count = 0

        for review in action_reviews:
            label = review.get("label", "REASONABLE")

            if label not in ["HIGH_YIELD", "LOW_YIELD", "INEFFICIENT", "CRITICAL_ERROR"]:
                continue

            memory = self._build_memory(case_result, review)

            if not self._is_duplicate(memory):
                self.memories.append(memory)
                added_count += 1

        return added_count

    def _build_memory(
        self,
        case_result: Dict[str, Any],
        review: Dict[str, Any],
    ) -> Dict[str, Any]:
        label = review.get("label", "REASONABLE")
        action_type = review.get("action_type", "")
        content = review.get("content", "")
        reason = review.get("reason", "")
        suggestion = review.get("suggestion", "")

        chief_complaint = case_result.get("chief_complaint", "")
        gold_diagnosis = case_result.get("gold_diagnosis", "")
        final_diagnosis = case_result.get("final_diagnosis", "")

        trigger_keywords = self._infer_keywords(
            chief_complaint=chief_complaint,
            gold_diagnosis=gold_diagnosis,
            final_diagnosis=final_diagnosis,
            content=content,
        )

        if label == "HIGH_YIELD":
            lesson_type = "positive"
            lesson = (
                f"在主诉为“{chief_complaint}”、最终诊断相关于“{gold_diagnosis}”的病例中，"
                f"{action_type} 动作“{content}”被评为 HIGH_YIELD。"
                f"经验：类似场景中可以优先考虑这类动作。"
            )
        else:
            lesson_type = "negative"
            lesson = (
                f"在主诉为“{chief_complaint}”、最终诊断相关于“{gold_diagnosis}”的病例中，"
                f"{action_type} 动作“{content}”被评为 {label}。"
                f"经验：类似场景中应谨慎使用或避免这类动作。"
            )

        return {
            "memory_id": self._make_memory_id(case_result, review),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "case_id": case_result.get("case_id", ""),
            "chief_complaint": chief_complaint,
            "gold_diagnosis": gold_diagnosis,
            "final_diagnosis": final_diagnosis,
            "turn": review.get("turn", ""),
            "action_type": action_type,
            "content": content,
            "label": label,
            "lesson_type": lesson_type,
            "lesson": lesson,
            "grader_reason": reason,
            "grader_suggestion": suggestion,
            "trigger_keywords": trigger_keywords,
        }

    def _make_memory_id(
        self,
        case_result: Dict[str, Any],
        review: Dict[str, Any],
    ) -> str:
        case_id = case_result.get("case_id", "unknown_case")
        turn = review.get("turn", "unknown_turn")
        label = review.get("label", "unknown_label")
        action_type = review.get("action_type", "unknown_action")
        content = review.get("content", "")

        content_hash = abs(hash(content)) % 100000

        return f"{case_id}_turn{turn}_{action_type}_{label}_{content_hash}"

    def _is_duplicate(self, memory: Dict[str, Any]) -> bool:
        memory_id = memory.get("memory_id")
        return any(item.get("memory_id") == memory_id for item in self.memories)

    def _infer_keywords(
        self,
        chief_complaint: str,
        gold_diagnosis: str,
        final_diagnosis: str,
        content: str,
    ) -> List[str]:
        text = f"{chief_complaint} {gold_diagnosis} {final_diagnosis} {content}"

        keyword_groups = {
            "肺炎": ["发热", "咳嗽", "咳痰", "胸片", "肺炎", "感染", "CRP", "血常规"],
            "心梗": ["胸痛", "心电图", "肌钙蛋白", "心肌梗死", "心梗", "ST段"],
            "糖尿病": ["多饮", "多尿", "体重下降", "血糖", "糖化血红蛋白", "糖尿病"],
            "阑尾炎": ["右下腹", "腹痛", "麦氏点", "反跳痛", "阑尾炎", "腹部超声"],
            "肺栓塞": ["呼吸困难", "胸痛", "咯血", "D-二聚体", "CTPA", "下肢", "肺栓塞"],
        }

        keywords = []

        for topic, terms in keyword_groups.items():
            if any(term in text for term in terms):
                keywords.append(topic)
                keywords.extend([term for term in terms if term in text])

        # 去重并保持顺序
        seen = set()
        unique_keywords = []

        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        return unique_keywords

    def _get_embedding_model(self):
        """
        懒加载 embedding 模型。
        只有 retrieval_mode='embedding' 时才会真正加载。
        """
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

        return self.embedding_model

    def _make_memory_embedding_signature(
            self,
            candidate_memories: List[Dict[str, Any]],
    ) -> str:
        """
        根据当前候选 memory 生成一个签名。
        如果签名不变，说明候选 memory 没变，可以复用 embedding 缓存。
        """
        memory_ids = [
            str(memory.get("memory_id", ""))
            for memory in candidate_memories
        ]

        return self.embedding_model_name + "||" + "||".join(memory_ids)

    def _get_memory_vectors_cached(
            self,
            candidate_memories: List[Dict[str, Any]],
    ):
        """
        获取候选 memory 的 embedding。
        如果候选 memory 没变，则直接复用缓存。
        """
        import numpy as np

        signature = self._make_memory_embedding_signature(candidate_memories)

        if (
                self._memory_embedding_cache_signature == signature
                and self._memory_embedding_cache_vectors is not None
        ):
            print("[Memory DEBUG] 使用缓存的 memory embeddings。")
            return self._memory_embedding_cache_vectors

        print(f"[Memory DEBUG] 重新计算 memory embeddings，数量={len(candidate_memories)}")

        model = self._get_embedding_model()

        memory_texts = [
            self._memory_to_retrieval_text(memory)
            for memory in candidate_memories
        ]

        memory_vecs = model.encode(
            memory_texts,
            normalize_embeddings=True,
        )

        memory_vecs = np.asarray(memory_vecs, dtype=np.float32)

        self._memory_embedding_cache_signature = signature
        self._memory_embedding_cache_vectors = memory_vecs

        return memory_vecs

    def _build_query_text(
            self,
            chief_complaint: str,
            transcript: List[Dict[str, Any]],
    ) -> str:
        query_text = f"患者主诉：{chief_complaint}\n"

        for item in transcript:
            query_text += f"动作：{item.get('action_type', '')} "
            query_text += f"内容：{item.get('content', '')} "
            query_text += f"观察结果：{item.get('observation', '')}\n"

        return query_text.strip()

    def _memory_to_retrieval_text(self, memory: Dict[str, Any]) -> str:
        """
        把一条 memory 转成用于 embedding 的文本。
        """
        parts = [
            f"主诉：{memory.get('chief_complaint', '')}",
            f"标准诊断：{memory.get('gold_diagnosis', '')}",
            f"Agent诊断：{memory.get('final_diagnosis', '')}",
            f"动作类型：{memory.get('action_type', '')}",
            f"动作内容：{memory.get('content', '')}",
            f"标签：{memory.get('label', '')}",
            f"经验类型：{memory.get('lesson_type', '')}",
            f"经验内容：{memory.get('lesson', '')}",
            f"评分理由：{memory.get('grader_reason', '')}",
            f"改进建议：{memory.get('grader_suggestion', '')}",
            f"关键词：{', '.join(memory.get('trigger_keywords', []))}",
        ]

        return "\n".join(parts)

    def _filter_memories_by_lesson_type(
            self,
            lesson_type_filter: str,
    ) -> List[Dict[str, Any]]:
        filtered = []

        for memory in self.memories:
            lesson_type = memory.get("lesson_type", "")

            if lesson_type_filter == "positive" and lesson_type != "positive":
                continue

            if lesson_type_filter == "negative" and lesson_type != "negative":
                continue

            filtered.append(memory)

        return filtered

    def retrieve_relevant_memories(
            self,
            chief_complaint: str,
            transcript: List[Dict[str, Any]],
            top_k: int = 5,
            lesson_type_filter: str = "all",
            retrieval_mode: str | None = None,
            similarity_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:

        """
        根据当前主诉和历史轨迹检索相关经验。

        lesson_type_filter:
        - "all": 正向和负向经验都检索
        - "positive": 只检索 HIGH_YIELD 正向经验
        - "negative": 只检索 LOW_YIELD / INEFFICIENT / CRITICAL_ERROR 负向经验

        retrieval_mode:
        - "keyword": 关键词检索
        - "embedding": 语义向量检索
        """
        mode = retrieval_mode or self.retrieval_mode

        if mode == "embedding":
            try:
                return self._retrieve_relevant_memories_embedding(
                    chief_complaint=chief_complaint,
                    transcript=transcript,
                    top_k=top_k,
                    lesson_type_filter=lesson_type_filter,
                    similarity_threshold=similarity_threshold,
                )

            except Exception as e:
                print(f"[Memory DEBUG] embedding 检索失败，回退到 keyword。错误：{e}")

        return self._retrieve_relevant_memories_keyword(
            chief_complaint=chief_complaint,
            transcript=transcript,
            top_k=top_k,
            lesson_type_filter=lesson_type_filter,
        )

    def _retrieve_relevant_memories_keyword(
            self,
            chief_complaint: str,
            transcript: List[Dict[str, Any]],
            top_k: int = 5,
            lesson_type_filter: str = "all",
    ) -> List[Dict[str, Any]]:
        query_text = self._build_query_text(chief_complaint, transcript)

        candidate_memories = self._filter_memories_by_lesson_type(lesson_type_filter)

        scored = []

        for memory in candidate_memories:
            score = 0

            for keyword in memory.get("trigger_keywords", []):
                if keyword and keyword in query_text:
                    score += 2

            for word in ["发热", "咳嗽", "胸痛", "呼吸困难", "腹痛", "多饮", "多尿"]:
                if word in query_text and word in memory.get("chief_complaint", ""):
                    score += 1

            if score > 0:
                scored.append((score, memory))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [memory for score, memory in scored[:top_k]]

    def _retrieve_relevant_memories_embedding(
            self,
            chief_complaint: str,
            transcript: List[Dict[str, Any]],
            top_k: int = 5,
            lesson_type_filter: str = "all",
            similarity_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        candidate_memories = self._filter_memories_by_lesson_type(lesson_type_filter)

        if not candidate_memories:
            return []

        model = self._get_embedding_model()

        query_text = self._build_query_text(chief_complaint, transcript)

        query_vec = model.encode(
            query_text,
            normalize_embeddings=True,
        )

        query_vec = np.asarray(query_vec, dtype=np.float32)

        memory_vecs = self._get_memory_vectors_cached(candidate_memories)

        scores_array = np.dot(memory_vecs, query_vec)
        scores = scores_array.tolist()

        scored = []

        for score, memory in zip(scores, candidate_memories):
            score = float(score)

            # 低于相似度阈值的不注入
            if score < similarity_threshold:
                continue

            memory_copy = dict(memory)
            memory_copy["retrieval_score"] = score
            scored.append((score, memory_copy))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [memory for score, memory in scored[:top_k]]

    def format_relevant_memories(
            self,
            chief_complaint: str,
            transcript: List[Dict[str, Any]],
            top_k: int = 5,
            lesson_type_filter: str = "all",
            retrieval_mode: str | None = None,
            similarity_threshold: float = 0.35,
            prompt_style: str = "full",
    ) -> str:

        mode = retrieval_mode or self.retrieval_mode

        memories = self.retrieve_relevant_memories(
            chief_complaint=chief_complaint,
            transcript=transcript,
            top_k=top_k,
            lesson_type_filter=lesson_type_filter,
            retrieval_mode=mode,
            similarity_threshold=similarity_threshold,
        )

        if not memories:
            return (
                f"暂无相关历史经验。"
                f"当前记忆模式：{lesson_type_filter}，检索方式：{mode}，格式：{prompt_style}"
            )

        if prompt_style == "compact":
            return self._format_memories_compact(
                memories=memories,
                lesson_type_filter=lesson_type_filter,
                retrieval_mode=mode,
            )

        if prompt_style == "balanced":
            return self._format_memories_balanced(
                memories=memories,
                lesson_type_filter=lesson_type_filter,
                retrieval_mode=mode,
            )

        return self._format_memories_full(
            memories=memories,
            lesson_type_filter=lesson_type_filter,
            retrieval_mode=mode,
        )

    def _format_memories_full(
            self,
            memories: List[Dict[str, Any]],
            lesson_type_filter: str,
            retrieval_mode: str,
    ) -> str:
        lines = []

        lines.append(f"当前记忆检索方式：{retrieval_mode}")
        lines.append(f"当前经验类型过滤：{lesson_type_filter}")
        lines.append("")

        for idx, memory in enumerate(memories, start=1):
            lines.append(f"经验 {idx}:")
            lines.append(f"- 类型：{memory.get('lesson_type')}")
            lines.append(f"- 标签：{memory.get('label')}")

            if "retrieval_score" in memory:
                lines.append(f"- 检索相似度：{memory.get('retrieval_score'):.4f}")

            lines.append(f"- 适用线索：{', '.join(memory.get('trigger_keywords', []))}")
            lines.append(f"- 经验内容：{memory.get('lesson')}")
            lines.append(f"- Grader理由：{memory.get('grader_reason')}")
            lines.append(f"- 改进建议：{memory.get('grader_suggestion')}")
            lines.append("")

        return "\n".join(lines).strip()

    def _format_memories_compact(
            self,
            memories: List[Dict[str, Any]],
            lesson_type_filter: str,
            retrieval_mode: str,
    ) -> str:
        """
        压缩版 memory prompt。
        只保留最关键的信息，减少 token。
        """
        lines = []

        lines.append(f"记忆检索方式：{retrieval_mode}")
        lines.append(f"经验过滤：{lesson_type_filter}")
        lines.append("以下是与当前病例相关的简短经验，请参考但不要盲目照搬。")
        lines.append("")

        for idx, memory in enumerate(memories, start=1):
            lesson_type = memory.get("lesson_type", "")
            label = memory.get("label", "")
            action_type = memory.get("action_type", "")
            content = memory.get("content", "")
            diagnosis = memory.get("gold_diagnosis", "")
            score_text = ""

            if "retrieval_score" in memory:
                score_text = f"，相似度={memory.get('retrieval_score'):.3f}"

            if lesson_type == "positive":
                prefix = "优先考虑"
            else:
                prefix = "谨慎避免"

            lines.append(
                f"{idx}. [{lesson_type}/{label}{score_text}] "
                f"场景≈{diagnosis}；{prefix}：{action_type}({content})。"
            )

        return "\n".join(lines).strip()

    def _format_memories_balanced(
            self,
            memories: List[Dict[str, Any]],
            lesson_type_filter: str,
            retrieval_mode: str,
    ) -> str:
        """
        折中版 memory prompt。
        比 full 短，但保留动作价值原因和改进建议。
        """
        lines = []

        lines.append(f"记忆检索方式：{retrieval_mode}")
        lines.append(f"经验过滤：{lesson_type_filter}")
        lines.append(
            "以下是与当前病例相关的经验。请优先参考 HIGH_YIELD 经验，避免 LOW_YIELD / INEFFICIENT / CRITICAL_ERROR 经验，但不要盲目照搬。")
        lines.append("")

        for idx, memory in enumerate(memories, start=1):
            lesson_type = memory.get("lesson_type", "")
            label = memory.get("label", "")
            action_type = memory.get("action_type", "")
            content = memory.get("content", "")
            diagnosis = memory.get("gold_diagnosis", "")
            reason = memory.get("grader_reason", "")
            suggestion = memory.get("grader_suggestion", "")

            score_text = ""
            if "retrieval_score" in memory:
                score_text = f"，相似度={memory.get('retrieval_score'):.3f}"

            if lesson_type == "positive":
                action_instruction = "可优先考虑"
            else:
                action_instruction = "应谨慎或避免"

            lines.append(f"经验 {idx} [{lesson_type}/{label}{score_text}]")
            lines.append(f"- 适用场景：{diagnosis}")
            lines.append(f"- 动作建议：{action_instruction} {action_type}({content})")
            lines.append(f"- 原因：{reason}")

            if suggestion and suggestion != "无需改进":
                lines.append(f"- 改进建议：{suggestion}")

            lines.append("")

        return "\n".join(lines).strip()

    def count(self) -> int:
        return len(self.memories)
