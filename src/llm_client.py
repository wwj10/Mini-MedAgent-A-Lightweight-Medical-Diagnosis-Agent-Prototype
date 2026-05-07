# import os
# from pathlib import Path
# from typing import List, Dict
#
# from dotenv import load_dotenv
# from openai import OpenAI
#
#
# class LLMClient:
#     """
#     统一管理大模型调用。
#     目前使用 OpenAI-compatible API，因此 DeepSeek 也可以通过 OpenAI SDK 调用。
#     """
#
#     def __init__(self):
#         project_root = Path(__file__).resolve().parents[1]
#         env_path = project_root / ".env"
#         load_dotenv(env_path)
#
#         api_key = os.getenv("OPENAI_API_KEY")
#         base_url = os.getenv("OPENAI_BASE_URL")
#         model_name = os.getenv("MODEL_NAME")
#
#         if not api_key:
#             raise ValueError("没有找到 OPENAI_API_KEY，请检查 .env 文件。")
#
#         if not base_url:
#             raise ValueError("没有找到 OPENAI_BASE_URL，请检查 .env 文件。")
#
#         if not model_name:
#             raise ValueError("没有找到 MODEL_NAME，请检查 .env 文件。")
#
#         self.model_name = model_name
#
#         self.client = OpenAI(
#             api_key=api_key,
#             base_url=base_url,
#         )
#
#     def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=messages,
#             temperature=temperature,
#         )
#
#         return response.choices[0].message.content
#     # def chat(self, messages, temperature: float = 0.2) -> str:
#     #     print("\n[LLM DEBUG] 正在调用大模型...")
#     #     print("[LLM DEBUG] model =", self.model_name)
#     #
#     #     response = self.client.chat.completions.create(
#     #     model=self.model_name,
#     #     messages=messages,
#     #     temperature=temperature,
#     # )
#     #
#     #     print("[LLM DEBUG] response id =", getattr(response, "id", None))
#     #     print("[LLM DEBUG] response model =", getattr(response, "model", None))
#     #     print("[LLM DEBUG] usage =", getattr(response, "usage", None))
#     #
#     #     return response.choices[0].message.content
import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    """
    统一管理大模型调用。
    目前使用 OpenAI-compatible API，因此 DeepSeek 也可以通过 OpenAI SDK 调用。
    同时记录调用次数和 token 用量，方便确认是否真的调用了 API。
    """

    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]
        env_path = project_root / ".env"
        load_dotenv(env_path)

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = os.getenv("MODEL_NAME")

        if not api_key:
            raise ValueError("没有找到 OPENAI_API_KEY，请检查 .env 文件。")

        if not base_url:
            raise ValueError("没有找到 OPENAI_BASE_URL，请检查 .env 文件。")

        if not model_name:
            raise ValueError("没有找到 MODEL_NAME，请检查 .env 文件。")

        self.model_name = model_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        print("\n[LLM DEBUG] 正在调用大模型...")
        print("[LLM DEBUG] model =", self.model_name)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )

        self.call_count += 1

        usage = getattr(response, "usage", None)

        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0

            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens

            print("[LLM DEBUG] prompt_tokens =", prompt_tokens)
            print("[LLM DEBUG] completion_tokens =", completion_tokens)
            print("[LLM DEBUG] total_tokens =", total_tokens)
        else:
            print("[LLM DEBUG] usage 信息为空，可能该平台未返回 token 统计。")

        return response.choices[0].message.content

    def get_usage(self) -> Dict[str, int | str]:
        return {
            "model_name": self.model_name,
            "call_count": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
