# src/config.py
"""클래스 10종 정의. 노트북과 웹앱(docs/classes.js)이 모두 여기서 가져간다."""
import json
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class MissionClass:
    id: int                 # YOLO 클래스 번호 (0~9)
    key: str                # 영문 키 (폴더 이름, 코드용)
    ko: str                 # 한국어 이름
    emoji: str
    oi_labels: tuple        # 이 클래스로 합칠 Open Images 라벨들
    coco_id: int | None     # COCO에 원래 있던 클래스면 그 번호 (기준선 평가용)


CLASSES = [
    MissionClass(0, "mouse",        "마우스", "🖱️", ("Computer mouse",), 64),
    MissionClass(1, "umbrella",     "우산",   "☂️", ("Umbrella",), 25),
    MissionClass(2, "book",         "책",     "📖", ("Book",), 73),
    MissionClass(3, "toothbrush",   "칫솔",   "🪥", ("Toothbrush",), 79),
    MissionClass(4, "hat",          "모자",   "🧢", ("Hat", "Fedora", "Sun hat", "Cowboy hat", "Sombrero"), None),
    MissionClass(5, "door",         "문",     "🚪", ("Door",), None),
    MissionClass(6, "toilet_paper", "휴지",   "🧻", ("Toilet paper",), None),
    MissionClass(7, "sock",         "양말",   "🧦", ("Sock",), None),
    MissionClass(8, "shoe",         "신발",   "👟", ("Footwear", "Boot", "High heels", "Sandal"), None),
    MissionClass(9, "glasses",      "안경",   "👓", ("Glasses", "Sunglasses"), None),
]
KEYS = [c.key for c in CLASSES]
BY_KEY = {c.key: c for c in CLASSES}
OLD = [c for c in CLASSES if c.coco_id is not None]     # COCO가 원래 알던 4종
NEW = [c for c in CLASSES if c.coco_id is None]         # 새로 가르치는 6종
OI_TO_KEY = {lab: c.key for c in CLASSES for lab in c.oi_labels}


def has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있으면 True (한글 유니코드: (code-0xAC00) % 28 != 0)."""
    code = ord(word[-1]) - 0xAC00
    return 0 <= code <= 11171 and code % 28 != 0


def mission_text(c: MissionClass) -> str:
    return f"{c.emoji} {c.ko}{'을' if has_batchim(c.ko) else '를'} 비춰라"


# ---------- v2 (실험 E): 13종 ----------
# 기존 0~9는 그대로 두고 뒤에 붙인다. CLASSES(10종)는 예전 노트북이 쓰므로 바꾸지 않는다.
# Open Images에는 "Tumbler"가 없어서 가장 가까운 "Bottle"을 텀블러로 쓴다.
CLASSES13 = CLASSES + [
    MissionClass(10, "cup",     "컵",     "☕", ("Mug", "Coffee cup"), None),
    MissionClass(11, "tumbler", "텀블러", "🥤", ("Bottle",), None),
    MissionClass(12, "chair",   "의자",   "🪑", ("Chair",), None),
]
KEYS13 = [c.key for c in CLASSES13]
BY_KEY13 = {c.key: c for c in CLASSES13}
OI_TO_KEY13 = {lab: c.key for c in CLASSES13 for lab in c.oi_labels}

# 하드 네거티브: 사람들이 카메라에 비출 법한 '미션이 아닌' 물건 + 손. 이 라벨의 박스는 내보내지 않는다(배경).
NEG_LABELS = ("Mobile phone", "Remote control", "Computer keyboard", "Human hand", "Pen", "Watch",
              "Headphones", "Tablet computer", "Spoon", "Box", "Pillow", "Towel", "Plastic bag",
              "Bowl", "Laptop", "Kitchen utensil")


def oi_to_key(classes=None) -> dict:
    """Open Images 라벨 → 클래스 키 (classes 기본값 = 10종)."""
    return {lab: c.key for c in (classes or CLASSES) for lab in c.oi_labels}


def to_js(classes=None, conf=None) -> str:
    """웹앱에서 import할 classes.js 내용. conf: {클래스 키: 신뢰도 기준}이 있으면 클래스마다 conf 칸을 넣는다."""
    rows = []
    for c in classes or CLASSES:
        row = {"id": c.id, "key": c.key, "ko": c.ko, "emoji": c.emoji}
        if conf and c.key in conf:
            row["conf"] = round(float(conf[c.key]), 2)
        rows.append(row)
    return "export const CLASSES = " + json.dumps(rows, ensure_ascii=False, indent=1) + ";\n"
