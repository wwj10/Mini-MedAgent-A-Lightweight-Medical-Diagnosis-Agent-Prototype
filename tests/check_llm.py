import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("MODEL_NAME")

if not api_key:
    raise ValueError("没有找到 OPENAI_API_KEY，请检查 .env 文件。")

if not model_name:
    raise ValueError("没有找到 MODEL_NAME，请检查 .env 文件。")

client_kwargs = {"api_key": api_key}

if base_url:
    client_kwargs["base_url"] = base_url

client = OpenAI(**client_kwargs)

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "你是一个严谨的医学AI Agent学习助手。"},
        {"role": "user", "content": "用一句话解释什么是医学 Agent。"}
    ],
    temperature=0.2,
)

print(response.choices[0].message.content)
