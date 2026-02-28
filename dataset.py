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

current_count = 0
if os.path.exists("dataset.jsonl"):
    with open("dataset.jsonl", "r", encoding="utf-8") as f:
        current_count = sum(1 for _ in f)

print(f"현재 저장된 데이터: {current_count}")

while current_count < TARGET_SIZE:
    try:
        batch = generate_batch()

        if isinstance(batch, list):
            with open("dataset.jsonl", "a", encoding="utf-8") as f:
                for item in batch:
                    if current_count >= TARGET_SIZE:
                        break
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    current_count += 1

            print(f"{current_count} / {TARGET_SIZE} 저장 완료")

        else:
            print("배열 아님, 재시도")

        time.sleep(SLEEP_SECONDS)

    except Exception as e:
        print("재시도:", e)
        time.sleep(SLEEP_SECONDS)

print("완료! dataset.jsonl 생성 완료.")
