from google import genai
import os
import random
import json
import time
import re
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

relationships = ["회사", "친구", "연인", "고객", "가족"]
intents = ["일정 조율", "거절", "사과", "확인 요청", "감사 표현", "정보 전달"]
tones = ["존댓말", "반말", "공손", "단호"]

TARGET_SIZE = 5
BATCH_SIZE = 5
SLEEP_SECONDS = 15

dataset = []

def clean_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    return text

def generate_batch():
    prompt = f"""
설명 없이 JSON 배열만 출력하세요.
총 {BATCH_SIZE}개의 서로 다른 데이터를 생성하세요.

각 항목 형식:
{{
  "conversation": ["상대: ...", "나: ..."],
  "relationship": "{random.choice(relationships)}",
  "intent": "{random.choice(intents)}",
  "tone": "{random.choice(tones)}",
  "response": "..."
}}

반드시 배열 형태로만 출력하세요.
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    text = clean_json(response.text)
    return json.loads(text)

while len(dataset) < TARGET_SIZE:
    try:
        batch = generate_batch()

        if isinstance(batch, list):
            dataset.extend(batch)
            print(f"{len(dataset)} / {TARGET_SIZE}")
        else:
            print("배열 아님, 재시도")

        time.sleep(SLEEP_SECONDS)

    except Exception as e:
        print("재시도:", e)
        time.sleep(SLEEP_SECONDS)

dataset = dataset[:TARGET_SIZE]

with open("dataset.jsonl", "a", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("완료! dataset.jsonl 생성됨.")
