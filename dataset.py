from google import genai
from google.genai import errors
import os
import random
import json
import time
import re
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# --- NEW: AUTO-DISCOVERY LOGIC ---
def get_available_flash_model():
    try:
        for m in client.models.list():
            # Look for flash models specifically
            if 'flash' in m.name.lower():
                # Strip 'models/' prefix if the SDK adds it automatically
                model_id = m.name.replace('models/', '')
                print(f"Using discovered model: {model_id}")
                return model_id
    except Exception as e:
        print(f"Could not list models: {e}")
    return "gemini-1.5-flash" # Fallback if list fails

SELECTED_MODEL = get_available_flash_model()
# --------------------------------

emotions = ["형식적", "상냥", "서운함", "단호", "친근", "분노"]
closeness_levels = [0, 1, 2, 3, 4]
TARGET_SIZE = 100
SLEEP_SECONDS = 15 

def clean_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    return text

def generate_single_item():
    emotion = random.choice(emotions)
    closeness = random.choice(closeness_levels)

    prompt = f"""
출력은 반드시 JSON 코드 블록 없이 순수 JSON 문자열만 출력하세요.
[조건] 감정: {emotion}, 친밀도: {closeness}
[출력 형식]
{{
  "context": ["상대: ...", "나: ...", "상대: ..."],
  "emotion": "{emotion}",
  "closeness": {closeness},
  "response": "..."
}}
"""
    response = client.models.generate_content(
        model=SELECTED_MODEL, 
        contents=prompt
    )
    
    if not response.text:
        raise ValueError("Empty response. Possibly blocked by safety filters.")
        
    return json.loads(clean_json(response.text))

current_count = 0
if os.path.exists("dataset.jsonl"):
    with open("dataset.jsonl", "r", encoding="utf-8") as f:
        current_count = sum(1 for _ in f)

print(f"시작 위치: {current_count} / {TARGET_SIZE}")

while current_count < TARGET_SIZE:
    try:
        item = generate_single_item()

        with open("dataset.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            current_count += 1

        print(f"[{current_count}/{TARGET_SIZE}] 저장 완료")
        time.sleep(SLEEP_SECONDS)

    except errors.ClientError as e:
        if "429" in str(e):
            print("Quota exceeded. Sleeping 30s...")
            time.sleep(30)
        elif "404" in str(e):
            print(f"Fatal Error: Model {SELECTED_MODEL} not found. Check API key permissions.")
            break
        else:
            print(f"API Error: {e}")
            time.sleep(5)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(2)

print("프로세스 종료.")
