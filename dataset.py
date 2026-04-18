from google import genai
from google.genai import errors
import os
import json
import time
import re
import random
from collections import defaultdict
from dotenv import load_dotenv

# =============================
# ENV / CLIENT
# =============================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY가 .env에 설정되어 있지 않습니다.")

client = genai.Client(api_key=api_key)


def get_available_flash_model():
    try:
        for m in client.models.list():
            if "flash" in m.name.lower():
                model_id = m.name.replace("models/", "")
                print(f"Using discovered model: {model_id}")
                return model_id
    except Exception as e:
        print(f"Could not list models: {e}")
    return "gemini-1.5-flash"


SELECTED_MODEL = get_available_flash_model()
JUDGE_MODEL = SELECTED_MODEL

# =============================
# LABEL DEFINITIONS
# =============================
EMOTION_DEFINITIONS = {
    "형식적": "존댓말, 감정 표현 최소, 거리감 유지",
    "서운함": "아쉬움, 속상함, 약한 감정 표현",
    "단호": "경계, 거절, 명확한 의사 표현",
    "친근": "편한 말투 우선, 가까운 관계의 자연스러운 표현",
    "분노": "강한 불쾌감, 공격적이되 과도한 욕설은 제외"
}

CLOSENESS_DEFINITIONS = {
    0: "완전 공식적, 일반적 존댓말 고정",
    1: "약한 거리감",
    2: "무난한 사이",
    3: "꽤 가까움",
    4: "매우 가까움"
}

ALLOWED_EMOTIONS_BY_CLOSENESS = {
    0: ["형식적"],
    1: ["서운함", "단호", "친근"],
    2: ["서운함", "단호", "친근"],
    3: ["서운함", "단호", "친근", "분노"],
    4: ["서운함", "단호", "친근", "분노"]
}

ALL_COMBOS = [
    (0, "형식적"),
    (1, "서운함"), (1, "단호"), (1, "친근"),
    (2, "서운함"), (2, "단호"), (2, "친근"),
    (3, "서운함"), (3, "단호"), (3, "친근"), (3, "분노"),
    (4, "서운함"), (4, "단호"), (4, "친근"), (4, "분노")
]

# =============================
# CONFIG
# =============================
TARGET_SIZE = 209
OUTPUT_FILE = "dataset.jsonl"
REJECT_FILE = "rejects.jsonl"
MAX_RETRIES_PER_ITEM = 4
SLEEP_SECONDS = 1
USE_JUDGE = True

GENERATION_CONFIG = {
    "response_mime_type": "application/json",
    "temperature": 0.95,
    "top_p": 0.95,
}

JUDGE_CONFIG = {
    "response_mime_type": "application/json",
    "temperature": 0.1,
}

# =============================
# TEXT RULES
# =============================
BANNED_WORDS = [
    "씨발", "병신", "좆", "개새끼", "꺼져", "죽어", "미친놈", "미친년"
]

ANGER_HEAVY_WORDS = [
    "어림없어", "웃기지 마", "짜증나", "열받", "진짜 기분 나쁘네",
    "됐거든", "말이 돼?", "장난해?", "너무한 거 아니야", "개빡", "빡치", "화나네"
]

DISAPPOINTED_HINTS = [
    "서운", "아쉽", "속상", "섭섭", "좀 그렇", "마음이 안 좋", "기분이 좀"
]

FIRM_HINTS = [
    "안 돼", "안돼", "안 될", "안될",
    "어렵", "힘들", "곤란", "불편",
    "못 해", "못하", "못 할", "못할",
    "이번에는 안", "이번엔 안",
    "그만해", "그만했으면", "그만해줘",
    "정리해줘", "선 지켜", "선을 지켜",
    "싫어", "원하지 않", "부담", "하지 말아",
    "안 했으면", "하지 않았으면", "자제해",
    "무리", "그건 아니", "거기까지", "더는", "더이상", "더 이상",
    "안 받", "안 빌려", "안 줄", "거절", "사양", "안 할게", "안할게"
]

FRIENDLY_HINTS = [
    "ㅋㅋ", "ㅎㅎ", "아 뭐야", "야", "나중에", "그러자", "알겠어", "오케이", "괜찮아"
]

FORMAL_ENDINGS = [
    "요", "니다", "드립니다", "해주세요", "부탁드립니다",
    "어렵습니다", "죄송합니다", "감사합니다", "괜찮습니다"
]

BANMAL_HINTS = [
    "야", "너", "어림없어", "됐어", "안 돼", "뭐야", "하자", "줄게", "싫어"
]

# =============================
# UTILS
# =============================
def clean_json(text: str) -> str:
    text = re.sub(r"```json|```", "", text).strip()
    return text


def make_sample_id(n: int) -> str:
    return f"sample_{n:04d}"


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def count_existing_lines(filepath: str) -> int:
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_existing_items(filepath: str):
    items = []
    if not os.path.exists(filepath):
        return items
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def make_dedup_key(item: dict) -> str:
    context_text = " || ".join(
        f"{turn['speaker']}:{normalize_text(turn['text'])}" for turn in item["context"]
    )
    response_text = normalize_text(item["response"])
    return f"{item['emotion']}|{item['closeness']}|{context_text}|{response_text}"


def combo_key(closeness: int, emotion: str) -> str:
    return f"{closeness}|{emotion}"


def write_jsonl(filepath: str, item: dict):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_politeness_score(text: str) -> int:
    """
    아주 단순한 휴리스틱:
    +1: 존댓말 종결/표현
    -1: 반말/가까운 표현
    """
    score = 0
    if any(x in text for x in ["습니다", "드릴게요", "해주세요", "죄송", "감사합니다", "괜찮습니다"]):
        score += 1
    if any(x in text for x in ["야", "너", "줄게", "됐어", "뭐야", "싫어"]):
        score -= 1
    return score


def is_firm_expression(text: str) -> bool:
    text = normalize_text(text)
    explicit_patterns = [
        "안 돼", "안돼", "안 될 것 같", "안될 것 같",
        "못 해", "못하", "못 할 것 같", "못할 것 같",
        "어려울 것 같", "어렵겠", "힘들 것 같", "곤란",
        "그만해", "그만해줘", "그만했으면",
        "하지 말아", "하지마", "안 했으면", "하지 않았으면",
        "부담돼", "부담된다", "불편해", "불편하다",
        "더는", "더 이상", "선 지켜", "선을 지켜",
        "싫어", "원하지 않아", "사양할게", "거절할게",
        "이번엔 안", "이번에는 안", "안 빌려", "못 빌려", "안 줄게",
        "어렵다", "힘들다", "곤란하다", "무리야", "무리다"
    ]
    return any(p in text for p in explicit_patterns)


# =============================
# PROMPTS
# =============================
def build_prompt(sample_id: str, emotion: str, closeness: int, attempt: int = 1) -> str:
    allowed_emotions = ", ".join(ALLOWED_EMOTIONS_BY_CLOSENESS[closeness])

    prompt = f"""
출력은 반드시 JSON 코드 블록 없이 "순수 JSON 문자열만" 출력하세요.
설명, 주석, 해설, 마크다운은 절대 포함하지 마세요.

당신의 역할은 한국어 메신저 답장 데이터 1개를 생성하는 것입니다.

[입력 조건]
- id: {sample_id}
- 감정: {emotion}
- 친밀도: {closeness}
- 상대방 유형은 따로 입력하지 않으며, 문맥(context) 안에서 자연스럽게 드러나도록 작성하세요.
- 친밀도에 따라 감정 표현 강도를 조절하세요.

[감정 라벨 정의]
- 형식적: {EMOTION_DEFINITIONS["형식적"]}
- 서운함: {EMOTION_DEFINITIONS["서운함"]}
- 단호: {EMOTION_DEFINITIONS["단호"]}
- 친근: {EMOTION_DEFINITIONS["친근"]}
- 분노: {EMOTION_DEFINITIONS["분노"]}

[친밀도 정의]
- 0: {CLOSENESS_DEFINITIONS[0]}
- 1: {CLOSENESS_DEFINITIONS[1]}
- 2: {CLOSENESS_DEFINITIONS[2]}
- 3: {CLOSENESS_DEFINITIONS[3]}
- 4: {CLOSENESS_DEFINITIONS[4]}

[허용 감정 규칙]
- 친밀도 {closeness}에서 허용되는 감정은 다음뿐입니다: {allowed_emotions}
- 현재 반드시 감정 "{emotion}" 으로 생성하세요.

[추가 규칙]
- 친밀도 0에서는 반말 금지
- 분노는 친밀도 3 이상에서만 허용
- 친근은 과도한 예의 표현보다 편한 말투 우선
- 서운함은 분노처럼 공격적으로 쓰지 않음
- 욕설, 혐오 표현, 과도한 비난 금지
- 실제 메신저처럼 자연스럽고 짧게 작성
- context는 2~4개의 발화로 구성
- response는 마지막 context 발화에 대한 user의 답장 1~2문장
- context의 마지막 발화는 반드시 speaker가 "other" 여야 함
- 상대방 유형(교수님, 부모님, 친구, 연인, 상사 등)이 문맥에서 유추 가능하도록 작성
- context는 반드시 아래 형식을 따르세요:
  - 각 발화는 객체여야 함
  - speaker 값은 "other" 또는 "user"만 허용
  - text 값에는 해당 발화 내용만 넣기
- response에는 speaker 정보 넣지 말고 답장 내용만 작성
- 같은 상황이 반복되지 않도록 다양한 주제를 사용하세요 (돈, 약속, 과제, 연락, 부탁, 일정, 감정 갈등 등)
- 너무 뻔한 패턴 문장을 반복하지 마세요
- response는 한국어 화자가 실제 메신저에서 보낼 법한 자연스러운 문장이어야 합니다.

- 단호는 단순히 차가운 분위기가 아니라, 거절/경계/중단 요청이 문장에 분명히 드러나야 합니다.
- 단호 response에는 "안 돼", "어렵다", "못 하겠다", "그만해줘", "부담된다"처럼 명확한 제한 표현이 포함되도록 하세요.

[출력 JSON 형식]
{{
  "id": "{sample_id}",
  "context": [
    {{"speaker": "other", "text": "..." }},
    {{"speaker": "user", "text": "..." }},
    {{"speaker": "other", "text": "..." }}
  ],
  "emotion": "{emotion}",
  "closeness": {closeness},
  "response": "..."
}}
""".strip()

    if emotion == "단호" and attempt >= 2:
        prompt += """

[재강조: 단호]
- 이번 샘플의 response는 반드시 명확한 거절, 경계, 선 긋기 중 하나를 포함해야 합니다.
- 우회적인 말만 하지 말고, 불가능/거절 의사를 분명하게 드러내세요.
"""

    if emotion == "단호" and attempt >= 3:
        prompt += """
- response 안에 다음과 같은 표현 유형 중 하나를 반드시 포함하세요:
  "안 돼", "어렵다", "못 하겠다", "그만해줘", "부담된다"
"""

    return prompt


def build_judge_prompt(item: dict) -> str:
    return f"""
당신은 한국어 메신저 답장 데이터의 라벨 품질을 검수하는 심사자입니다.
설명 없이 반드시 순수 JSON만 출력하세요.

[라벨 정의]
- 형식적: 존댓말, 감정 표현 최소, 거리감 유지
- 서운함: 아쉬움, 속상함, 약한 감정 표현
- 단호: 경계, 거절, 명확한 의사 표현
- 친근: 편한 말투 우선, 가까운 관계의 자연스러운 표현
- 분노: 강한 불쾌감, 공격적이되 과도한 욕설은 제외

[중요 판정 기준]
1. 주어진 emotion 라벨과 실제 response의 톤이 맞아야 한다.
2. closeness=0이면 반드시 존댓말 위주여야 한다.
3. 단호는 선 긋기/거절은 분명하지만 과도한 공격성은 없어야 한다.
4. 분노는 강한 불쾌감이 드러나야 하지만 욕설은 금지다.
5. 서운함은 공격적이면 안 된다.
6. 친근은 부드럽고 편한 말투여야 한다.
7. context와 response가 자연스럽게 이어져야 한다.

반드시 아래 형식으로만 출력:
{{
  "is_valid": true,
  "predicted_emotion": "단호",
  "reason": "간단한 이유"
}}

검수 대상:
{json.dumps(item, ensure_ascii=False)}
""".strip()


# =============================
# BASIC VALIDATION
# =============================
def validate_item(item: dict) -> bool:
    if not isinstance(item, dict):
        return False

    required_keys = {"id", "context", "emotion", "closeness", "response"}
    if not required_keys.issubset(item.keys()):
        return False

    if not isinstance(item["id"], str) or not item["id"].startswith("sample_"):
        return False

    if not isinstance(item["context"], list):
        return False
    if len(item["context"]) < 2 or len(item["context"]) > 4:
        return False

    for turn in item["context"]:
        if not isinstance(turn, dict):
            return False
        if "speaker" not in turn or "text" not in turn:
            return False
        if turn["speaker"] not in {"other", "user"}:
            return False
        if not isinstance(turn["text"], str) or not turn["text"].strip():
            return False

    if item["context"][-1]["speaker"] != "other":
        return False

    emotion = item["emotion"]
    closeness = item["closeness"]

    if closeness not in ALLOWED_EMOTIONS_BY_CLOSENESS:
        return False
    if emotion not in ALLOWED_EMOTIONS_BY_CLOSENESS[closeness]:
        return False

    if not isinstance(item["response"], str) or not item["response"].strip():
        return False

    return True


# =============================
# SEMANTIC VALIDATION
# =============================
def semantic_validate(item: dict) -> tuple[bool, str]:
    response = normalize_text(item["response"])
    emotion = item["emotion"]
    closeness = item["closeness"]

    if contains_any(response, BANNED_WORDS):
        return False, "욕설/금지어 포함"

    if len(response) < 4:
        return False, "response가 너무 짧음"

    if len(response) > 120:
        return False, "response가 너무 김"

    # closeness 0: 존댓말 위주
    if closeness == 0:
        if contains_any(response, ["야", "너", "어림없어", "됐어", "뭐야", "줄게", "싫어"]):
            return False, "친밀도 0인데 반말/거친 표현 포함"
        if extract_politeness_score(response) <= 0 and not any(response.endswith(e) or e in response for e in FORMAL_ENDINGS):
            return False, "친밀도 0인데 형식적 존댓말 부족"

    # 형식적
    if emotion == "형식적":
        if contains_any(response, BANMAL_HINTS):
            return False, "형식적 라벨인데 반말/가까운 표현 포함"
        if contains_any(response, ANGER_HEAVY_WORDS):
            return False, "형식적 라벨인데 공격성 표현 포함"

    # 서운함
    if emotion == "서운함":
        if contains_any(response, ANGER_HEAVY_WORDS):
            return False, "서운함 라벨인데 분노 표현이 과함"
        if not (contains_any(response, DISAPPOINTED_HINTS) or "좀" in response or "괜히" in response):
            return False, "서운함의 감정 신호가 부족"

    # 단호
    if emotion == "단호":
        if contains_any(response, ANGER_HEAVY_WORDS):
            return False, "단호 라벨인데 분노 표현이 과함"
        if not is_firm_expression(response):
            return False, "단호 라벨인데 거절/경계 표현이 부족"

    # 친근
    if emotion == "친근":
        if closeness == 0:
            return False, "친근은 closeness 0에서 허용되지 않음"
        if contains_any(response, ANGER_HEAVY_WORDS):
            return False, "친근 라벨인데 공격적 표현 포함"
        if closeness >= 2 and any(response.endswith(x) for x in ["습니다", "드립니다", "죄송합니다"]):
            return False, "친근 라벨인데 지나치게 형식적"
        if len(response.split()) <= 2:
            return False, "친근 라벨인데 표현이 너무 건조함"

    # 분노
    if emotion == "분노":
        if closeness < 3:
            return False, "분노는 closeness 3 이상만 허용"
        if not contains_any(response, ANGER_HEAVY_WORDS + ["기분 나빠", "불쾌", "화나", "그만 좀 해", "짜증", "열받"]):
            return False, "분노 라벨인데 강한 불쾌감 표현이 부족"

    # response에 구조 정보 실수 방지
    if "speaker" in response or '"text"' in response:
        return False, "response에 구조 정보가 섞임"

    return True, "ok"


def llm_judge(item: dict) -> tuple[bool, str]:
    if not USE_JUDGE:
        return True, "judge skipped"

    prompt = build_judge_prompt(item)

    try:
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=JUDGE_CONFIG
        )
        if not response.text:
            return False, "judge empty response"

        raw = clean_json(response.text)
        result = json.loads(raw)

        is_valid = bool(result.get("is_valid", False))
        predicted = result.get("predicted_emotion", "")
        reason = result.get("reason", "")

        if not is_valid:
            return False, f"judge reject: predicted={predicted}, reason={reason}"

        if predicted != item["emotion"]:
            return False, f"judge emotion mismatch: predicted={predicted}, expected={item['emotion']}"

        return True, f"judge ok: {reason}"

    except Exception as e:
        return False, f"judge exception: {e}"


# =============================
# GENERATION
# =============================
def generate_one_item(sample_id: str, emotion: str, closeness: int, attempt: int = 1) -> dict:
    prompt = build_prompt(sample_id, emotion, closeness, attempt=attempt)

    response = client.models.generate_content(
        model=SELECTED_MODEL,
        contents=prompt,
        config=GENERATION_CONFIG
    )

    if not response.text:
        raise ValueError("Empty response")

    raw = clean_json(response.text)
    item = json.loads(raw)
    if not isinstance(item, dict):
        raise ValueError("Generated output is not a dict")

    return item


# =============================
# DATASET STATE
# =============================
def load_dataset_state(filepath: str):
    items = load_existing_items(filepath)
    seen_keys = set()
    combo_counts = defaultdict(int)
    max_id_num = 0

    for item in items:
        if not validate_item(item):
            continue

        k = make_dedup_key(item)
        seen_keys.add(k)
        combo_counts[combo_key(item["closeness"], item["emotion"])] += 1

        m = re.match(r"sample_(\d+)", item["id"])
        if m:
            max_id_num = max(max_id_num, int(m.group(1)))

    return {
        "items": items,
        "seen_keys": seen_keys,
        "combo_counts": combo_counts,
        "max_id_num": max_id_num
    }


def build_target_plan(target_size: int):
    base = target_size // len(ALL_COMBOS)
    remainder = target_size % len(ALL_COMBOS)

    targets = {}
    for idx, (cl, em) in enumerate(ALL_COMBOS):
        targets[combo_key(cl, em)] = base + (1 if idx < remainder else 0)
    return targets


def choose_next_combo(combo_counts: dict, combo_targets: dict):
    deficits = []
    for cl, em in ALL_COMBOS:
        k = combo_key(cl, em)
        remain = combo_targets[k] - combo_counts.get(k, 0)
        if remain > 0:
            deficits.append((remain, cl, em))

    if not deficits:
        return None

    deficits.sort(reverse=True)
    top_remain = deficits[0][0]
    candidates = [(cl, em) for remain, cl, em in deficits if remain == top_remain]
    return random.choice(candidates)


# =============================
# SAVE / REJECT
# =============================
def save_item(item: dict, state: dict):
    write_jsonl(OUTPUT_FILE, item)
    state["items"].append(item)
    state["seen_keys"].add(make_dedup_key(item))
    state["combo_counts"][combo_key(item["closeness"], item["emotion"])] += 1


def save_reject(sample_id: str, closeness: int, emotion: str, reason: str, raw_item=None):
    record = {
        "id": sample_id,
        "closeness": closeness,
        "emotion": emotion,
        "reason": reason,
        "raw_item": raw_item
    }
    write_jsonl(REJECT_FILE, record)


# =============================
# MAIN
# =============================
def main():
    state = load_dataset_state(OUTPUT_FILE)
    combo_targets = build_target_plan(TARGET_SIZE)

    saved_count = len(state["items"])
    next_id_num = state["max_id_num"] + 1

    print(f"Model: {SELECTED_MODEL}")
    print(f"현재 저장 수: {saved_count}/{TARGET_SIZE}")
    print("조합별 목표:")
    for cl, em in ALL_COMBOS:
        k = combo_key(cl, em)
        print(f"  closeness={cl}, emotion={em}: {state['combo_counts'].get(k, 0)}/{combo_targets[k]}")

    while saved_count < TARGET_SIZE:
        combo = choose_next_combo(state["combo_counts"], combo_targets)
        if combo is None:
            print("모든 조합 목표를 채웠습니다.")
            break

        closeness, emotion = combo
        sample_id = make_sample_id(next_id_num)
        next_id_num += 1

        print(f"\n[{saved_count}/{TARGET_SIZE}] 생성 시도 -> id={sample_id}, closeness={closeness}, emotion={emotion}")

        success = False

        for attempt in range(1, MAX_RETRIES_PER_ITEM + 1):
            try:
                item = generate_one_item(sample_id, emotion, closeness, attempt=attempt)

                # 기본 검증
                if not validate_item(item):
                    save_reject(sample_id, closeness, emotion, f"basic validate fail (attempt {attempt})", item)
                    print(f"  - basic validate fail (attempt {attempt})")
                    time.sleep(SLEEP_SECONDS)
                    continue

                # id / label 정합성
                if item["id"] != sample_id:
                    save_reject(sample_id, closeness, emotion, f"id mismatch: {item['id']} != {sample_id}", item)
                    print(f"  - id mismatch (attempt {attempt})")
                    time.sleep(SLEEP_SECONDS)
                    continue

                if item["closeness"] != closeness or item["emotion"] != emotion:
                    save_reject(sample_id, closeness, emotion, "label mismatch", item)
                    print(f"  - label mismatch (attempt {attempt})")
                    time.sleep(SLEEP_SECONDS)
                    continue

                # normalize
                item["response"] = normalize_text(item["response"])
                for turn in item["context"]:
                    turn["text"] = normalize_text(turn["text"])

                # dedup
                dkey = make_dedup_key(item)
                if dkey in state["seen_keys"]:
                    save_reject(sample_id, closeness, emotion, f"duplicate sample (attempt {attempt})", item)
                    print(f"  - duplicate (attempt {attempt})")
                    time.sleep(SLEEP_SECONDS)
                    continue

                # semantic validate
                sem_ok, sem_reason = semantic_validate(item)

                # judge fallback 허용
                judge_ok = False
                judge_reason = "judge not run"

                if sem_ok:
                    judge_ok, judge_reason = llm_judge(item)
                    if not judge_ok:
                        save_reject(sample_id, closeness, emotion, f"judge fail: {judge_reason}", item)
                        print(f"  - judge fail: {judge_reason} (attempt {attempt})")
                        time.sleep(SLEEP_SECONDS)
                        continue
                else:
                    judge_ok, judge_reason = llm_judge(item)
                    if judge_ok:
                        print(f"  - semantic fail but judge passed: {sem_reason} / {judge_reason}")
                    else:
                        save_reject(sample_id, closeness, emotion, f"semantic fail: {sem_reason} / {judge_reason}", item)
                        print(f"  - semantic fail: {sem_reason} (attempt {attempt})")
                        time.sleep(SLEEP_SECONDS)
                        continue

                # save
                save_item(item, state)
                saved_count += 1
                success = True
                print(f"  ✓ saved ({saved_count}/{TARGET_SIZE})")
                break

            except errors.ClientError as e:
                if "429" in str(e):
                    print("  - 429 quota exceeded, 60초 대기")
                    time.sleep(60)
                else:
                    print(f"  - API client error: {e}")
                    time.sleep(10)
            except Exception as e:
                print(f"  - Unexpected error: {e}")
                time.sleep(5)

        if not success:
            save_reject(sample_id, closeness, emotion, "max retries exceeded", None)
            print(f"  ✗ failed permanently: {sample_id}")

        time.sleep(SLEEP_SECONDS)

    print("\n최종 완료")
    print(f"저장 파일: {OUTPUT_FILE}")
    print(f"리젝트 파일: {REJECT_FILE}")

    print("\n최종 조합 분포:")
    for cl, em in ALL_COMBOS:
        k = combo_key(cl, em)
        print(f"  closeness={cl}, emotion={em}: {state['combo_counts'].get(k, 0)}/{combo_targets[k]}")


if __name__ == "__main__":
    main()
