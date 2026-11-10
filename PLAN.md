# "○○을 비춰라" 기상 미션 알람 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open Images 10종으로 YOLO11n을 파인튜닝하고, 아이폰 Safari에서 "○○을 비춰라" 미션을 통과해야 꺼지는 알람 웹앱을 GitHub Pages로 배포한다.

**Architecture:** 파이썬 쪽은 `src/`의 작은 모듈(설정 · 데이터 · 후처리 · 평가)과, 이것들을 불러 쓰는 노트북 한 개(생성기 `tools/build_nb.py`로 만듦)로 나눈다. 웹 쪽은 `docs/`에 정적 파일만 두고, 후처리(`postprocess.js`)는 순수 함수로 분리해 node로 테스트한다. 파이썬 `src/postprocess.py`는 JS와 같은 동작을 하는 쌍둥이 구현이며, ONNX 검증에서 "웹이 할 계산"을 파이썬으로 미리 확인하는 데 쓴다.

**Tech Stack:** Python 3.12 (uv venv) · ultralytics (YOLO11n) · fiftyone (Open Images V7) · onnx / onnxruntime · pytest · 바닐라 JS + onnxruntime-web 1.20.1 (CDN) · node:test · GitHub Pages

**기준 문서:** `SPEC.md`

---

## 파일 구조

```
wakeup_mission/
├── SPEC.md, PLAN.md
├── .gitignore                ← .venv, data/, runs/, *.pt, my_photos/
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py             ← 클래스 10종, Open Images 라벨 합치기 표, COCO id, 한국어 이름 (단일 진실 공급원)
│   ├── data.py               ← Open Images 받기 → YOLO 형식 내보내기, 기준선용 val 부분집합 만들기
│   ├── postprocess.py        ← decode / nms / pick_main / scan_crop (docs/postprocess.js와 같은 동작)
│   └── evaluate.py           ← 클래스별 지표 표, 직접 찍은 사진 미션 통과율
├── tests/
│   ├── test_config.py
│   ├── test_postprocess.py
│   └── postprocess.test.mjs  ← node --test
├── tools/build_nb.py         ← wakeup_mission.ipynb 생성
├── wakeup_mission.ipynb
├── docs/                     ← GitHub Pages 배포 폴더
│   ├── index.html, style.css, app.js
│   ├── postprocess.js        ← 순수 함수 (테스트 대상)
│   ├── classes.js            ← 노트북이 config.py에서 생성
│   └── model.onnx            ← 노트북이 생성
├── data/  runs/  my_photos/<클래스키>/*.jpg
```

---

### Task 1: 프로젝트 뼈대 · 가상환경

**Files:**
- Create: `wakeup_mission/.gitignore`, `wakeup_mission/requirements.txt`, `wakeup_mission/src/__init__.py`

- [ ] **Step 1: 파일 만들기**

`.gitignore`
```
.venv/
data/
runs/
my_photos/
*.pt
__pycache__/
.ipynb_checkpoints/
.DS_Store
```

`requirements.txt`
```
ultralytics
fiftyone
onnx
onnxruntime
onnxslim
pandas
matplotlib
pytest
ipykernel
nbformat
nbconvert
```

`src/__init__.py` — 빈 파일.

- [ ] **Step 2: 가상환경 + 커널 등록**

```bash
cd wakeup_mission
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name wakeup --display-name "wakeup (.venv)"
.venv/bin/python -c "import ultralytics, fiftyone, onnxruntime, torch; print(ultralytics.__version__, torch.backends.mps.is_available())"
```
Expected: 버전 출력과 `True`.

- [ ] **Step 3: git 시작 + 커밋**

```bash
git init -b main
git add .gitignore requirements.txt src/__init__.py SPEC.md PLAN.md
git commit -m "chore: 프로젝트 뼈대"
```

---

### Task 2: `src/config.py` — 클래스 정의 (TDD)

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 실패하는 테스트**

```python
# tests/test_config.py
from src import config as C

def test_ten_classes_in_order():
    assert [c.key for c in C.CLASSES] == [
        "mouse", "umbrella", "book", "toothbrush",
        "hat", "door", "toilet_paper", "sock", "shoe", "glasses"]
    assert [c.id for c in C.CLASSES] == list(range(10))

def test_every_class_has_open_images_label():
    for c in C.CLASSES:
        assert c.oi_labels, c.key
    # 합치기 표: Open Images 라벨 → 우리 클래스 키
    assert C.OI_TO_KEY["Fedora"] == "hat"
    assert C.OI_TO_KEY["Sunglasses"] == "glasses"
    assert C.OI_TO_KEY["High heels"] == "shoe"
    assert set(C.OI_TO_KEY.values()) == {c.key for c in C.CLASSES}

def test_old_classes_have_coco_ids():
    assert {c.key: c.coco_id for c in C.CLASSES if c.coco_id is not None} == {
        "mouse": 64, "umbrella": 25, "book": 73, "toothbrush": 79}
    assert [c.key for c in C.OLD] == ["mouse", "umbrella", "book", "toothbrush"]

def test_mission_text_uses_right_particle():
    assert C.mission_text(C.BY_KEY["book"]) == "📖 책을 비춰라"
    assert C.mission_text(C.BY_KEY["mouse"]) == "🖱️ 마우스를 비춰라"

def test_js_export_contains_all_classes():
    js = C.to_js()
    assert js.startswith("export const CLASSES = [")
    assert js.count('"key"') == 10
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_config.py -q`
Expected: FAIL (`ImportError: cannot import name 'config'`)

- [ ] **Step 3: 구현**

```python
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


def to_js() -> str:
    """웹앱에서 import할 classes.js 내용."""
    rows = [{"id": c.id, "key": c.key, "ko": c.ko, "emoji": c.emoji} for c in CLASSES]
    return "export const CLASSES = " + json.dumps(rows, ensure_ascii=False, indent=1) + ";\n"
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_config.py -q`
Expected: `5 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: 클래스 10종 설정"
```

---

### Task 3: `src/postprocess.py` — YOLO 출력 해석 (TDD)

YOLO11 ONNX 출력은 `[1, 4+nc, N]`. 앞 4행은 박스 중심·크기(cx, cy, w, h, 입력 픽셀 단위), 나머지 nc행은 **이미 시그모이드를 거친** 클래스 점수다. 입력 320이면 N = 40² + 20² + 10² = 2100.

**Files:**
- Create: `src/postprocess.py`
- Test: `tests/test_postprocess.py`

- [ ] **Step 1: 실패하는 테스트**

```python
# tests/test_postprocess.py
import numpy as np
from src.postprocess import decode, nms, pick_main, scan_crop, iou

def make_output(boxes, nc=3, n=5):
    """boxes: [(i, cx, cy, w, h, cls, score)] → [1, 4+nc, n] 배열"""
    out = np.zeros((1, 4 + nc, n), dtype=np.float32)
    for i, cx, cy, w, h, cls, score in boxes:
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + cls, i] = score
    return out

def test_decode_keeps_confident_and_converts_to_corners():
    out = make_output([(1, 100, 100, 40, 20, 2, 0.9), (3, 50, 50, 10, 10, 0, 0.3)])
    dets = decode(out, conf=0.5)
    assert len(dets) == 1
    d = dets[0]
    assert (d["x1"], d["y1"], d["x2"], d["y2"]) == (80, 90, 120, 110)
    assert d["cls"] == 2 and abs(d["score"] - 0.9) < 1e-6

def test_iou():
    a = dict(x1=0, y1=0, x2=10, y2=10)
    b = dict(x1=5, y1=0, x2=15, y2=10)
    assert abs(iou(a, b) - 50 / 150) < 1e-9

def test_nms_removes_overlap_same_class_only():
    a = dict(x1=0, y1=0, x2=10, y2=10, score=0.9, cls=0)
    b = dict(x1=1, y1=0, x2=11, y2=10, score=0.8, cls=0)   # a와 많이 겹침 → 제거
    c = dict(x1=1, y1=0, x2=11, y2=10, score=0.7, cls=1)   # 다른 클래스 → 유지
    kept = nms([b, a, c], iou_thres=0.45)
    assert kept == [a, c]

def test_pick_main_largest_above_min_area():
    small = dict(x1=0, y1=0, x2=50, y2=50, score=0.9, cls=0)      # 2500/102400 = 2.4%
    big = dict(x1=0, y1=0, x2=200, y2=200, score=0.6, cls=1)      # 39%
    assert pick_main([small, big], input_size=320, min_area=0.15) == big
    assert pick_main([small], input_size=320, min_area=0.15) is None

def test_scan_crop_centered_square():
    assert scan_crop(1280, 720, ratio=0.7) == (388, 108, 504)
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_postprocess.py -q`
Expected: FAIL (`ModuleNotFoundError: src.postprocess`)

- [ ] **Step 3: 구현**

```python
# src/postprocess.py
"""YOLO ONNX 출력 후처리. docs/postprocess.js와 같은 동작이어야 한다 (웹앱이 할 계산을 파이썬으로 검증)."""
import numpy as np


def decode(output, conf=0.5):
    """[1, 4+nc, N] → 신뢰도 conf 이상인 박스 목록 (각 위치에서 점수가 가장 높은 클래스 하나)."""
    o = output[0]
    scores = o[4:]                       # [nc, N]
    cls = scores.argmax(0)
    best = scores.max(0)
    dets = []
    for i in np.where(best >= conf)[0]:
        cx, cy, w, h = o[:4, i]
        dets.append(dict(x1=float(cx - w / 2), y1=float(cy - h / 2),
                         x2=float(cx + w / 2), y2=float(cy + h / 2),
                         score=float(best[i]), cls=int(cls[i])))
    return dets


def iou(a, b):
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, iou_thres=0.45):
    """점수 높은 순으로 고르고, 같은 클래스끼리 많이 겹치는 박스는 버린다."""
    keep = []
    for d in sorted(dets, key=lambda d: d["score"], reverse=True):
        if all(k["cls"] != d["cls"] or iou(k, d) < iou_thres for k in keep):
            keep.append(d)
    return keep


def pick_main(dets, input_size, min_area=0.15):
    """스캔 틀(=입력 전체) 면적의 min_area 이상인 박스 중 가장 큰 것. 없으면 None."""
    best, best_area = None, 0.0
    for d in dets:
        area = (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]) / (input_size * input_size)
        if area >= min_area and area > best_area:
            best, best_area = d, area
    return best


def scan_crop(w, h, ratio=0.7):
    """프레임 가운데 정사각형 스캔 틀 (x, y, 한 변). 짧은 변의 ratio배."""
    s = round(ratio * min(w, h))
    return (w - s) // 2, (h - s) // 2, s
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_postprocess.py -q`
Expected: `5 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/postprocess.py tests/test_postprocess.py
git commit -m "feat: YOLO 출력 후처리 (파이썬)"
```

---

### Task 4: `docs/postprocess.js` — 같은 후처리 + 연속 판정 (TDD)

**Files:**
- Create: `docs/postprocess.js`
- Test: `tests/postprocess.test.mjs`

- [ ] **Step 1: 실패하는 테스트**

```js
// tests/postprocess.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { decode, iou, nms, pickMain, scanCrop, Streak } from '../docs/postprocess.js';

function makeOutput(boxes, nc = 3, n = 5) {
  const data = new Float32Array((4 + nc) * n);
  for (const [i, cx, cy, w, h, cls, score] of boxes) {
    data[i] = cx; data[n + i] = cy; data[2 * n + i] = w; data[3 * n + i] = h;
    data[(4 + cls) * n + i] = score;
  }
  return data;
}

test('decode: 신뢰도 거르기 + 모서리 좌표', () => {
  const data = makeOutput([[1, 100, 100, 40, 20, 2, 0.9], [3, 50, 50, 10, 10, 0, 0.3]]);
  const dets = decode(data, 3, 5, 0.5);
  assert.equal(dets.length, 1);
  const d = dets[0];
  assert.deepEqual([d.x1, d.y1, d.x2, d.y2, d.cls], [80, 90, 120, 110, 2]);
  assert.ok(Math.abs(d.score - 0.9) < 1e-6);
});

test('iou', () => {
  assert.ok(Math.abs(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 5, y1: 0, x2: 15, y2: 10 }) - 50 / 150) < 1e-9);
});

test('nms: 같은 클래스 겹침만 제거', () => {
  const a = { x1: 0, y1: 0, x2: 10, y2: 10, score: 0.9, cls: 0 };
  const b = { x1: 1, y1: 0, x2: 11, y2: 10, score: 0.8, cls: 0 };
  const c = { x1: 1, y1: 0, x2: 11, y2: 10, score: 0.7, cls: 1 };
  assert.deepEqual(nms([b, a, c], 0.45), [a, c]);
});

test('pickMain: 최소 면적 이상 중 가장 큰 박스', () => {
  const small = { x1: 0, y1: 0, x2: 50, y2: 50, score: 0.9, cls: 0 };
  const big = { x1: 0, y1: 0, x2: 200, y2: 200, score: 0.6, cls: 1 };
  assert.equal(pickMain([small, big], 320, 0.15), big);
  assert.equal(pickMain([small], 320, 0.15), null);
});

test('scanCrop: 가운데 정사각형', () => {
  assert.deepEqual(scanCrop(1280, 720, 0.7), [388, 108, 504]);
});

test('Streak: 연속 시간만큼 채우면 1, 짧은 끊김은 봐주고 긴 끊김은 초기화', () => {
  const s = new Streak(500, 250);
  assert.equal(s.update(true, 0), 0);
  assert.equal(s.update(true, 250), 0.5);
  assert.equal(s.update(false, 300), 0.5);   // 250ms 안의 끊김 → 진행 유지
  assert.equal(s.update(true, 500), 1);
  s.reset();
  s.update(true, 1000);
  assert.equal(s.update(false, 1300), 0);     // 300ms 끊김 → 초기화
});
```

- [ ] **Step 2: 실패 확인**

Run: `node --test tests/`
Expected: FAIL (`Cannot find module '../docs/postprocess.js'`)

- [ ] **Step 3: 구현**

```js
// docs/postprocess.js
// YOLO ONNX 출력 후처리. src/postprocess.py와 같은 동작.
// 출력 배열 모양: [1, 4+nc, n] → data[c * n + i] 가 i번째 위치의 c번째 값

export function decode(data, nc, n, conf) {
  const dets = [];
  for (let i = 0; i < n; i++) {
    let cls = 0, score = -1;
    for (let c = 0; c < nc; c++) {
      const s = data[(4 + c) * n + i];
      if (s > score) { score = s; cls = c; }
    }
    if (score < conf) continue;
    const cx = data[i], cy = data[n + i], w = data[2 * n + i], h = data[3 * n + i];
    dets.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score, cls });
  }
  return dets;
}

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

export function nms(dets, iouThres) {
  const keep = [];
  for (const d of [...dets].sort((p, q) => q.score - p.score)) {
    if (keep.every(k => k.cls !== d.cls || iou(k, d) < iouThres)) keep.push(d);
  }
  return keep;
}

export function pickMain(dets, inputSize, minArea) {
  let best = null, bestArea = 0;
  for (const d of dets) {
    const area = (d.x2 - d.x1) * (d.y2 - d.y1) / (inputSize * inputSize);
    if (area >= minArea && area > bestArea) { best = d; bestArea = area; }
  }
  return best;
}

export function scanCrop(w, h, ratio) {
  const s = Math.round(ratio * Math.min(w, h));
  return [Math.floor((w - s) / 2), Math.floor((h - s) / 2), s];
}

// 미션 물건이 needMs 동안 "연속으로" 보이면 진행률 1. graceMs 이하의 짧은 끊김(한두 프레임 놓침)은 봐준다.
export class Streak {
  constructor(needMs, graceMs) { this.needMs = needMs; this.graceMs = graceMs; this.reset(); }
  reset() { this.start = null; this.lastSeen = null; }
  update(seen, now) {
    if (seen) {
      if (this.start === null) this.start = now;
      this.lastSeen = now;
    } else if (this.lastSeen === null || now - this.lastSeen > this.graceMs) {
      this.reset();
      return 0;
    }
    return Math.min(1, ((this.lastSeen ?? now) - this.start) / this.needMs);
  }
}
```

- [ ] **Step 4: 통과 확인**

Run: `node --test tests/`
Expected: `pass 6`, `fail 0`

- [ ] **Step 5: 커밋**

```bash
git add docs/postprocess.js tests/postprocess.test.mjs
git commit -m "feat: YOLO 출력 후처리 + 연속 판정 (JS)"
```

---

### Task 5: `src/data.py` — Open Images 받기 · YOLO 내보내기

네트워크·대용량 다운로드라 단위 테스트 대신 **작은 실제 실행**으로 확인한다.

**Files:**
- Create: `src/data.py`

- [ ] **Step 1: 구현**

```python
# src/data.py
"""Open Images V7에서 10종을 받아 YOLO 형식으로 내보낸다."""
import os
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

from . import config as C


def load_split(split: str, per_class: int, seed: int = 51) -> fo.Dataset:
    """클래스마다 따로 받아(per_class장) 하나로 합친다. 같은 사진은 한 번만 들어간다."""
    name = f"wakeup-{split}-{per_class}"
    if fo.dataset_exists(name):
        return fo.load_dataset(name)
    merged = fo.Dataset(name, persistent=True)
    for c in C.CLASSES:
        part = foz.load_zoo_dataset(
            "open-images-v7", split=split, label_types=["detections"],
            classes=list(c.oi_labels), max_samples=per_class, seed=seed, shuffle=True,
            dataset_name=f"oi-{split}-{c.key}-{per_class}")
        merged.merge_samples(part)          # filepath 기준으로 합침 (중복 제거)
    return merged


def to_mission_view(ds: fo.Dataset):
    """10종에 해당하는 박스만 남기고, Open Images 라벨을 우리 클래스 키로 바꾼다."""
    view = ds.filter_labels("ground_truth", F("label").is_in(list(C.OI_TO_KEY)), only_matches=True)
    return view.map_labels("ground_truth", C.OI_TO_KEY)


def export_yolo(view, out_dir: Path, split: str):
    """YOLO 형식: out_dir/images/{split}, out_dir/labels/{split}, out_dir/dataset.yaml"""
    view.export(export_dir=str(out_dir), dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth", split=split, classes=C.KEYS)


def count_table(view):
    """클래스별 사진 수 · 박스 수 (pandas DataFrame)."""
    import pandas as pd
    boxes = view.count_values("ground_truth.detections.label")
    rows = []
    for c in C.CLASSES:
        n_img = len(view.match(F("ground_truth.detections").filter(F("label") == c.key).length() > 0))
        rows.append(dict(id=c.id, 클래스=c.ko, 키=c.key, 사진=n_img, 박스=boxes.get(c.key, 0),
                         구분="기존(COCO)" if c.coco_id is not None else "새로"))
    return pd.DataFrame(rows)


def make_old_subset(yolo_dir: Path, out_dir: Path):
    """기존 4종이 있는 val 사진만 모아 두 벌의 라벨을 만든다.
    - out_dir/ours : 우리 번호(0~3) → 파인튜닝 모델 평가용
    - out_dir/coco : COCO 번호(64, 25, 73, 79) → 사전학습 COCO YOLO 평가용
    두 벌이 같은 사진을 쓰므로 기준선과 공정하게 비교된다."""
    old_ids = {c.id: c.coco_id for c in C.OLD}
    coco_names = _coco_names()
    for variant in ("ours", "coco"):
        (out_dir / variant / "images" / "val").mkdir(parents=True, exist_ok=True)
        (out_dir / variant / "labels" / "val").mkdir(parents=True, exist_ok=True)
    for lab in sorted((yolo_dir / "labels" / "val").glob("*.txt")):
        rows = [r.split() for r in lab.read_text().splitlines() if r.strip()]
        rows = [r for r in rows if int(r[0]) in old_ids]
        if not rows:
            continue
        img = next((yolo_dir / "images" / "val").glob(lab.stem + ".*"))
        for variant, mapping in (("ours", {k: k for k in old_ids}), ("coco", old_ids)):
            dst = out_dir / variant / "images" / "val" / img.name
            if not dst.exists():
                try:
                    os.link(img, dst)           # 하드링크 (용량 안 씀)
                except OSError:
                    shutil.copy(img, dst)
            text = "\n".join(" ".join([str(mapping[int(r[0])])] + r[1:]) for r in rows) + "\n"
            (out_dir / variant / "labels" / "val" / lab.name).write_text(text)
    _write_yaml(out_dir / "ours.yaml", out_dir / "ours", C.KEYS)
    _write_yaml(out_dir / "coco.yaml", out_dir / "coco", coco_names)


def _coco_names():
    from ultralytics import YOLO
    names = YOLO("yolo11n.pt").names
    return [names[i] for i in range(len(names))]


def _write_yaml(path: Path, root: Path, names):
    lines = [f"path: {root.resolve()}", "train: images/val", "val: images/val", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 2: 작은 실행으로 확인 (클래스당 5장)**

```bash
.venv/bin/python -c "
from pathlib import Path
from src import data
v = data.to_mission_view(data.load_split('validation', 5))
print(data.count_table(v))
data.export_yolo(v, Path('data/smoke'), 'val')
print(open('data/smoke/dataset.yaml').read())
"
ls data/smoke/labels/val | head -3; head -3 "data/smoke/labels/val/$(ls data/smoke/labels/val | head -1)"
```
Expected: 10행 표(사진 수가 대부분 ≥1), `names:`에 0: mouse … 9: glasses, 라벨 줄이 `<0~9> cx cy w h` 형식.
처음 실행 시 FiftyOne이 주석 CSV를 받느라 몇 분 걸릴 수 있다.

- [ ] **Step 3: 확인용 데이터 지우고 커밋**

```bash
rm -rf data/smoke
.venv/bin/python -c "import fiftyone as fo; [fo.delete_dataset(n) for n in fo.list_datasets() if n.endswith('-5')]"
git add src/data.py
git commit -m "feat: Open Images 받기 + YOLO 내보내기"
```

---

### Task 6: `src/evaluate.py` — 지표 표 · 직접 찍은 사진 미션 통과율 (TDD 일부)

**Files:**
- Create: `src/evaluate.py`
- Modify: `tests/test_postprocess.py` (끝에 테스트 추가)

- [ ] **Step 1: 실패하는 테스트 (순수 함수 부분)**

`tests/test_postprocess.py` 끝에 추가:
```python
from src.evaluate import mission_passed

def test_mission_passed_needs_target_class_as_main_box():
    big_sock = dict(x1=0, y1=0, x2=200, y2=200, score=0.8, cls=7)
    big_shoe = dict(x1=0, y1=0, x2=250, y2=250, score=0.9, cls=8)
    assert mission_passed([big_sock], target=7, input_size=320, min_area=0.15)
    assert not mission_passed([big_sock, big_shoe], target=7, input_size=320, min_area=0.15)  # 더 큰 게 신발
    assert not mission_passed([], target=7, input_size=320, min_area=0.15)
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_postprocess.py -q`
Expected: FAIL (`ModuleNotFoundError: src.evaluate`)

- [ ] **Step 3: 구현**

```python
# src/evaluate.py
"""평가 도우미: 클래스별 지표 표, 직접 찍은 사진으로 미션 통과율."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .postprocess import decode, nms, pick_main, scan_crop

INPUT, CONF, IOU, MIN_AREA, SCAN_RATIO = 320, 0.5, 0.45, 0.15, 0.7   # 웹앱 app.js와 같은 값


def per_class_table(metrics, names) -> pd.DataFrame:
    """ultralytics val() 결과 → 클래스별 P, R, mAP50, mAP50-95 표."""
    b = metrics.box
    rows = []
    for j, ci in enumerate(b.ap_class_index):
        rows.append(dict(id=int(ci), 클래스=names[int(ci)], P=b.p[j], R=b.r[j], mAP50=b.ap50[j], **{"mAP50-95": b.ap[j]}))
    return pd.DataFrame(rows).round(3)


def mission_passed(dets, target, input_size=INPUT, min_area=MIN_AREA) -> bool:
    """웹앱 판정과 같은 규칙: 틀 안의 '주인공 박스'가 목표 클래스여야 통과."""
    main = pick_main(dets, input_size, min_area)
    return main is not None and main["cls"] == target


def preprocess(img_bgr):
    """웹앱과 같은 전처리: 가운데 스캔 틀 자르기 → 320 → RGB → 0~1 → [1,3,320,320]."""
    import cv2
    h, w = img_bgr.shape[:2]
    x, y, s = scan_crop(w, h, SCAN_RATIO)
    crop = cv2.resize(img_bgr[y:y + s, x:x + s], (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
    rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None].copy(), crop


def run_onnx(session, tensor):
    out = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    return nms(decode(out, CONF), IOU)


def my_photos_table(session, photo_dir: Path) -> pd.DataFrame:
    """my_photos/<클래스키>/*.jpg 를 미션으로 보고 통과 여부를 표로."""
    import cv2
    rows = []
    for c in C.CLASSES:
        for p in sorted((photo_dir / c.key).glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            tensor, _ = preprocess(cv2.imread(str(p)))
            dets = run_onnx(session, tensor)
            main = pick_main(dets, INPUT, MIN_AREA)
            rows.append(dict(미션=c.ko, 파일=p.name, 통과=mission_passed(dets, c.id),
                             잡힌것=C.CLASSES[main["cls"]].ko if main else "-",
                             점수=round(main["score"], 2) if main else None))
    return pd.DataFrame(rows)
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `11 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/evaluate.py tests/test_postprocess.py
git commit -m "feat: 평가 도우미 (지표 표, 미션 통과율)"
```

---

### Task 7: 노트북 생성기 `tools/build_nb.py`

수업 표현 방식, 마크다운으로 "왜"를 설명, 코드 주석은 적당히. `QUICK = True`면 클래스당 20장 · 2에폭으로 빠르게 끝까지 돈다(검증용).

**Files:**
- Create: `tools/build_nb.py`
- Create (생성됨): `wakeup_mission.ipynb`

- [ ] **Step 1: 생성기 작성**

```python
# tools/build_nb.py
"""wakeup_mission.ipynb 생성기. 실행: .venv/bin/python tools/build_nb.py"""
from pathlib import Path
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# ⏰ "○○을 비춰라" 기상 미션 알람 — YOLO11n 10종 파인튜닝

알람을 끄려면 **제시된 물건을 카메라에 비춰야** 한다. 아무 컵이나 되는 *일반 모델*이 필요하다.

| 단계 | 내용 |
|---|---|
| 1 | 클래스 10종 정하기 (COCO가 아는 4종 + 새로 6종) |
| 2 | Open Images V7에서 데이터 받기 → YOLO 형식 |
| 3 | 데이터 살펴보기 (개수, 예시, **빠진 라벨** 함정) |
| 4 | 모델 구조 (백본 · 넥 · 헤드, 무엇이 새로 초기화되나) |
| 5 | 기준선: COCO YOLO 그대로 기존 4종 |
| 6 | 실험 A 전체 미세조정 / 실험 B 백본 고정 |
| 7 | 평가: 클래스별 지표, 혼동행렬, 기준선 비교 |
| 8 | ONNX 내보내기 + 웹앱과 같은 계산으로 검증 |
| 9 | 직접 찍은 사진으로 미션 통과율 |
""")

md("## 0. 설정")
code("""
import sys, time, shutil, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from IPython.display import Image, display

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src import config as C, data as D, evaluate as E

QUICK = False                      # True: 클래스당 20장 · 2에폭 (끝까지 도는지만 확인)
PER_CLASS = {'train': 20 if QUICK else 400, 'validation': 10 if QUICK else 80}
EPOCHS = 2 if QUICK else 50
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
DATA = ROOT / 'data'
YOLO_DIR = DATA / ('yolo_quick' if QUICK else 'yolo')
RUNS = ROOT / 'runs'
plt.rcParams['font.family'] = 'AppleGothic'; plt.rcParams['axes.unicode_minus'] = False
print('device:', DEVICE, '| QUICK:', QUICK)
""")

md("## 1. 클래스 10종\n\nOpen Images는 라벨이 **계층 구조**다 (모자 ⊃ 페도라·솜브레로…). 하위 라벨을 상위 클래스로 합친다.")
code("""
pd.DataFrame([dict(id=c.id, 미션=C.mission_text(c), 키=c.key,
                   OpenImages라벨=', '.join(c.oi_labels),
                   COCO번호=c.coco_id if c.coco_id is not None else '-') for c in C.CLASSES])
""")

md("## 2. 데이터 받기 (Open Images V7 → YOLO 형식)\n\n클래스마다 따로 받아 합친다. 처음엔 주석 파일을 받느라 오래 걸린다. 한 번 받으면 캐시를 쓴다.")
code("""
views = {}
for split, yolo_split in [('train', 'train'), ('validation', 'val')]:
    ds = D.load_split(split, PER_CLASS[split])
    views[yolo_split] = D.to_mission_view(ds)
    if not (YOLO_DIR / 'labels' / yolo_split).exists():
        D.export_yolo(views[yolo_split], YOLO_DIR, yolo_split)
YAML = YOLO_DIR / 'dataset.yaml'
print(YAML.read_text())
""")

md("## 3. 데이터 살펴보기")
code("""
tables = {s: D.count_table(v).set_index('id') for s, v in views.items()}
summary = tables['train'][['클래스', '구분']].copy()
summary['train 사진'] = tables['train']['사진']; summary['train 박스'] = tables['train']['박스']
summary['val 사진'] = tables['val']['사진'];     summary['val 박스'] = tables['val']['박스']
summary
""")
code("""
# 클래스별 예시 한 장씩 (박스 그려서)
import cv2
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for ax, c in zip(axes.flat, C.CLASSES):
    lab = next(p for p in sorted((YOLO_DIR / 'labels' / 'train').glob('*.txt'))
               if any(r.split()[0] == str(c.id) for r in p.read_text().splitlines()))
    img_path = next((YOLO_DIR / 'images' / 'train').glob(lab.stem + '.*'))
    img = cv2.imread(str(img_path))[:, :, ::-1].copy()
    h, w = img.shape[:2]
    for r in lab.read_text().splitlines():
        k, cx, cy, bw, bh = r.split(); cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        color = (255, 60, 60) if int(k) == c.id else (60, 160, 255)
        cv2.rectangle(img, (int((cx - bw/2)*w), int((cy - bh/2)*h)), (int((cx + bw/2)*w), int((cy + bh/2)*h)), color, max(2, w // 200))
    ax.imshow(img); ax.set_title(C.mission_text(c)); ax.axis('off')
plt.tight_layout(); plt.show()
""")
md("""
### ⚠️ 빠진 라벨 함정

Open Images는 사진마다 **확인한 클래스만** 박스를 그린다. 아래는 "신발" 라벨이 있는 사진 중 **양말 라벨은 없는** 사진이다.
양말이 보이는데 라벨이 없다면, 모델은 그 양말을 **배경**으로 배운다 → 양말 Recall이 낮아질 수 있다.
""")
code("""
from fiftyone import ViewField as F
v = views['train']
has = lambda k: F('ground_truth.detections').filter(F('label') == k).length() > 0
shoe_no_sock = v.match(has('shoe') & ~has('sock')).take(6, seed=0)
fig, axes = plt.subplots(1, 6, figsize=(20, 4))
for ax, s in zip(axes, shoe_no_sock):
    ax.imshow(plt.imread(s.filepath)); ax.axis('off')
plt.suptitle('신발 라벨 O · 양말 라벨 X — 양말이 보이나?'); plt.show()
""")

md("""
## 4. 모델 구조 — YOLO11n

```
입력 640×640×3
 ▼ 백본: Conv → Conv → C3k2 → … → SPPF → C2PSA     ─ P3(80×80) · P4(40×40) · P5(20×20)
 ▼ 넥 (PAN-FPN): 위→아래, 아래→위로 크기가 다른 특징 섞기
 ▼ 헤드 (Detect, 크기 3개마다)
     ├ 박스 갈래:   4×16 채널 (DFL)
     └ 클래스 갈래: nc 채널  ← 80 → 10 으로 바뀌어 **이 층만 새로 초기화**
 ▼ 8,400개 위치 × [박스 4 + 클래스 10] → 신뢰도 거르기 → NMS
```

- **C3k2**: 입력을 나눠 일부만 합성곱 → 다시 합침 (잔차 연결과 비슷)
- **SPPF**: 맥스풀링을 겹쳐 넓은 범위를 봄 · **C2PSA**: 어텐션
- 클래스 점수는 softmax가 아니라 **클래스별 시그모이드(BCE)** → "신발 0.7, 양말 0.6"이 동시에 가능
""")
code("""
base = YOLO('yolo11n.pt')
base.info(detailed=False)
head = base.model.model[-1]
print('헤드 클래스 갈래 마지막 층:', [m[-1] for m in head.cv3])   # out_channels=80 → 파인튜닝하면 10
""")

md("## 5. 기준선 — COCO YOLO 그대로 (기존 4종만)\n\n기존 4종이 있는 val 사진만 모아, **같은 사진**으로 기준선과 파인튜닝 모델을 비교한다.")
code("""
OLD_DIR = DATA / ('val_old_quick' if QUICK else 'val_old')
if not (OLD_DIR / 'coco.yaml').exists():
    D.make_old_subset(YOLO_DIR, OLD_DIR)
coco_ids = [c.coco_id for c in C.OLD]
m0 = base.val(data=str(OLD_DIR / 'coco.yaml'), classes=coco_ids, imgsz=640, device=DEVICE,
              project=str(RUNS), name='baseline_val', exist_ok=True, plots=False, verbose=False)
baseline = E.per_class_table(m0, base.names)
baseline
""")

md("""
## 6. 학습 — 실험 A (전체 미세조정) vs 실험 B (백본 고정)

- **A**: 모든 층을 작은 학습률로 학습
- **B**: `freeze=10` → 백본(앞 10개 층)은 COCO 가중치 그대로, 넥·헤드만 학습

> "새 6종만 학습해서 기존 4종을 잊는지 보기"는 하지 않는다. 클래스 수가 바뀌면 클래스 층이 새로 초기화되므로,
> 그건 **잊은 게 아니라 배운 적이 없는 것**이다. 대신 기준선과 비교해 기존 4종이 나빠졌는지 본다.

Early Stopping: `patience=10` — 검증 fitness(mAP 기반 점수, 버전마다 식이 조금 다름)가 10에폭 동안 안 오르면 멈추고 `best.pt`를 남긴다.
""")
code("""
EXPS = {'A_full': dict(freeze=None), 'B_freeze10': dict(freeze=10)}
train_time = {}
for name, extra in EXPS.items():
    if (RUNS / name / 'weights' / 'best.pt').exists() and not QUICK:
        print(name, '이미 학습됨 → 건너뜀'); continue
    t0 = time.time()
    YOLO('yolo11n.pt').train(data=str(YAML), epochs=EPOCHS, imgsz=640, batch=16, patience=10,
                             device=DEVICE, project=str(RUNS), name=name, exist_ok=True,
                             seed=0, workers=4, plots=True, **extra)
    train_time[name] = time.time() - t0
train_time
""")
code("""
# 학습곡선 (ultralytics가 저장한 results.csv)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for name in EXPS:
    r = pd.read_csv(RUNS / name / 'results.csv'); r.columns = r.columns.str.strip()
    axes[0].plot(r['epoch'], r['train/box_loss'], label=f'{name} train'); axes[0].plot(r['epoch'], r['val/box_loss'], '--', label=f'{name} val')
    axes[1].plot(r['epoch'], r['train/cls_loss'], label=f'{name} train'); axes[1].plot(r['epoch'], r['val/cls_loss'], '--', label=f'{name} val')
    axes[2].plot(r['epoch'], r['metrics/mAP50(B)'], label=name)
for ax, t in zip(axes, ['box loss', 'cls loss', 'val mAP50']): ax.set_title(t); ax.legend(); ax.set_xlabel('epoch')
plt.tight_layout(); plt.show()
""")

md("## 7. 평가")
code("""
results, models = {}, {}
for name in EXPS:
    models[name] = YOLO(RUNS / name / 'weights' / 'best.pt')
    m = models[name].val(data=str(YAML), imgsz=640, device=DEVICE, project=str(RUNS),
                         name=f'{name}_val', exist_ok=True, plots=True, verbose=False)
    results[name] = E.per_class_table(m, C.KEYS).set_index('id')
    results[name].loc['전체', 'mAP50'] = m.box.map50
compare = pd.DataFrame({
    '클래스': [c.ko for c in C.CLASSES] + ['전체'],
    '구분': ['기존' if c.coco_id is not None else '새로' for c in C.CLASSES] + [''],
    **{f'{n} mAP50': list(results[n]['mAP50'].reindex(list(range(10)) + ['전체'])) for n in EXPS},
    **{f'{n} R': list(results[n]['R'].reindex(list(range(10)) + ['전체'])) for n in EXPS},
})
compare
""")
code("""
# 기존 4종: 기준선 vs A vs B (같은 사진)
old_rows = {'기준선(COCO)': list(baseline.set_index('id').loc[[c.coco_id for c in C.OLD], 'mAP50'])}   # COCO 번호 순서 → 우리 순서
for name in EXPS:
    m = models[name].val(data=str(OLD_DIR / 'ours.yaml'), classes=[c.id for c in C.OLD], imgsz=640,
                         device=DEVICE, project=str(RUNS), name=f'{name}_old', exist_ok=True, plots=False, verbose=False)
    old_rows[name] = list(E.per_class_table(m, C.KEYS)['mAP50'])
pd.DataFrame(old_rows, index=[c.ko for c in C.OLD])
""")
code("""
BEST = max(EXPS, key=lambda n: results[n].loc['전체', 'mAP50'])
print('앱에 쓸 모델:', BEST)
display(Image(str(RUNS / f'{BEST}_val' / 'confusion_matrix_normalized.png'), width=700))
""")
md("혼동행렬에서 **양말↔신발**, **안경↔배경**을 먼저 보자. 아래는 정답(왼쪽)과 예측(오른쪽) 예시.")
code("""
for kind in ['labels', 'pred']:
    display(Image(str(RUNS / f'{BEST}_val' / f'val_batch0_{kind}.jpg'), width=900))
""")

md("## 8. ONNX 내보내기 + 웹앱과 같은 계산으로 검증\n\n웹앱은 스캔 틀을 320×320으로 잘라 넣는다. 그래서 `imgsz=320`으로 내보낸다.")
code("""
import onnxruntime as ort
onnx_path = Path(models[BEST].export(format='onnx', imgsz=320, opset=12, simplify=True, dynamic=False))
sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
print('입력:', sess.get_inputs()[0].shape, '| 출력:', sess.get_outputs()[0].shape)   # [1,3,320,320] → [1,14,2100]
""")
code("""
# 같은 사진에 대해 ① ultralytics(PyTorch) 예측 ② 우리 전처리 + ONNX + 우리 후처리(src/postprocess.py) 비교
import cv2
checked = 0
for lab in sorted((YOLO_DIR / 'labels' / 'val').glob('*.txt'))[:30]:
    img = cv2.imread(str(next((YOLO_DIR / 'images' / 'val').glob(lab.stem + '.*'))))
    tensor, crop = E.preprocess(img)
    ours = E.run_onnx(sess, tensor)
    ref = models[BEST].predict(crop, imgsz=320, conf=E.CONF, iou=E.IOU, device='cpu', verbose=False)[0].boxes
    if len(ref) == 0 and len(ours) == 0: continue
    # 신뢰도 0.5 경계에 걸린 박스 하나 정도는 부동소수 차이로 갈릴 수 있다 → 1개 차이까지 허용
    assert abs(len(ours) - len(ref)) <= 1, (lab.name, len(ours), len(ref))
    if len(ref) and len(ours):
        top_ref = int(ref.cls[ref.conf.argmax()]); top_ours = max(ours, key=lambda d: d['score'])['cls']
        assert top_ref == top_ours, lab.name
    checked += 1
print(f'✅ {checked}장에서 박스 수(±1) · 최고 점수 클래스 일치')
""")
code("""
# 웹앱으로 복사
DOCS = ROOT / 'docs'
shutil.copy(onnx_path, DOCS / 'model.onnx')
(DOCS / 'classes.js').write_text(C.to_js())
print('docs/model.onnx', round((DOCS / 'model.onnx').stat().st_size / 1e6, 1), 'MB')
""")

md("## 9. 직접 찍은 사진으로 미션 통과율\n\n`my_photos/<클래스키>/*.jpg` 에 사진을 넣는다 (예: `my_photos/sock/1.jpg`). 웹앱과 **같은 규칙**(가운데 틀 · 면적 15% · 신뢰도 0.5)으로 판정한다.")
code("""
PHOTOS = ROOT / 'my_photos'
if PHOTOS.exists() and any(PHOTOS.rglob('*.jpg')):
    mine = E.my_photos_table(sess, PHOTOS)
    display(mine.groupby('미션')['통과'].agg(['mean', 'count']).rename(columns={'mean': '통과율', 'count': '장수'}))
    display(mine[~mine['통과']])
else:
    print('my_photos/ 에 사진을 넣고 다시 실행하세요.')
""")

md("""
## 10. 정리 — 스터디에서 이야기할 것

1. **사전학습 + 파인튜닝**: 클래스 층만 새로 만들고 나머지는 COCO에서 가져온다. 400장씩으로 새 물건 6종을 배운다.
2. **전체 미세조정 vs 백본 고정**: 위 표의 A/B 차이. 백본을 고정하면 빠르지만 새 물건을 덜 배우는가?
3. **기존 4종**: 기준선 대비 나빠졌나? (클래스 층이 새로 초기화됐는데도 400장으로 회복하는가)
4. **헷갈린 쌍**: 혼동행렬의 양말↔신발 · 빠진 라벨 함정
5. **시그모이드 vs softmax**: YOLO 클래스 점수는 서로 독립
6. **웹앱으로 옮기기**: 후처리(decode · NMS)를 직접 짜서 PyTorch와 같은 결과인지 확인했다
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "wakeup", "display_name": "wakeup (.venv)", "language": "python"}})
out = Path(__file__).resolve().parent.parent / "wakeup_mission.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
```

- [ ] **Step 2: 생성 + QUICK 실행으로 끝까지 도는지 확인**

```bash
.venv/bin/python tools/build_nb.py
# 노트북은 자기 폴더를 작업 폴더로 쓰므로, QUICK 사본도 같은 폴더에 만든다
sed 's/QUICK = False/QUICK = True/' wakeup_mission.ipynb > wm_quick.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute wm_quick.ipynb --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=wakeup
rm wm_quick.ipynb
```
Expected: 에러 없이 끝. 마지막 출력에 `✅ N장에서 … 일치`. (QUICK 모드는 `data/yolo_quick`, `runs/`를 씀. `docs/model.onnx`도 QUICK 모델로 생김 → 정식 실행이 덮어씀)

> QUICK에서 2에폭 모델은 박스를 거의 못 내서 `checked`가 0일 수 있다. 그래도 assert는 통과해야 한다.

- [ ] **Step 3: 커밋**

```bash
git add tools/build_nb.py wakeup_mission.ipynb
git commit -m "feat: 학습·평가 노트북"
```

---

### Task 8: 웹앱 화면 · 스타일

**Files:**
- Create: `docs/index.html`, `docs/style.css`

- [ ] **Step 1: `docs/index.html`**

```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <title>비춰라 알람</title>
  <link rel="stylesheet" href="style.css">
  <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"></script>
  <script type="module" src="app.js"></script>
</head>
<body>
  <section id="setup" class="screen active">
    <h1>⏰ 비춰라 알람</h1>
    <p class="sub">알람이 울리면 제시된 물건을 카메라에 비춰야 꺼져요.</p>
    <input id="time" type="time" value="07:00">
    <button id="arm" class="primary">알람 켜기</button>
    <button id="now">지금 미션 시작 (시연)</button>
    <p id="status" class="status">모델 준비 전</p>
  </section>

  <section id="armed" class="screen">
    <div class="big" id="armedTime">07:00</div>
    <p class="sub">에 울려요. 화면을 켜 둔 채로 두세요.</p>
    <button id="cancel">취소</button>
  </section>

  <section id="scan" class="screen">
    <video id="video" playsinline muted></video>
    <canvas id="overlay"></canvas>
    <div id="mission" class="banner"></div>
    <div id="message" class="message"></div>
  </section>

  <section id="done" class="screen">
    <div class="big">🎉 기상 성공!</div>
    <p class="sub" id="doneText"></p>
    <canvas id="snapshot"></canvas>
    <button id="again" class="primary">처음으로</button>
  </section>
</body>
</html>
```

- [ ] **Step 2: `docs/style.css`**

```css
:root { --bg: #0b0d12; --fg: #f4f6fb; --muted: #9aa3b5; --accent: #3ddc97; --warn: #ff6b6b; }
* { box-sizing: border-box; margin: 0; }
html, body { height: 100%; background: var(--bg); color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif; }
.screen { display: none; min-height: 100%; padding: 24px 16px; flex-direction: column;
  align-items: center; justify-content: center; gap: 16px; text-align: center; }
.screen.active { display: flex; }
h1 { font-size: 28px; }
.sub { color: var(--muted); }
.big { font-size: 56px; font-weight: 700; }
.status { color: var(--muted); font-size: 14px; }
input[type=time] { font-size: 40px; background: #1a1e28; color: var(--fg); border: 0; border-radius: 12px; padding: 8px 16px; }
button { width: min(320px, 100%); font-size: 18px; padding: 14px; border-radius: 12px; border: 0;
  background: #1f2430; color: var(--fg); }
button.primary { background: var(--accent); color: #062; font-weight: 700; }
#scan { padding: 0; position: fixed; inset: 0; }
#video, #overlay { position: absolute; inset: 0; width: 100%; height: 100%; }
#video { object-fit: cover; }
.banner { position: absolute; top: max(24px, env(safe-area-inset-top)); left: 16px; right: 16px;
  font-size: 26px; font-weight: 800; padding: 12px; border-radius: 14px; background: rgba(0,0,0,.6); }
.message { position: absolute; bottom: 120px; left: 16px; right: 16px; font-size: 20px; font-weight: 700; color: var(--warn); min-height: 28px; }
#snapshot { width: min(320px, 100%); border-radius: 12px; }
```

- [ ] **Step 3: 커밋**

```bash
git add docs/index.html docs/style.css
git commit -m "feat: 웹앱 화면"
```

---

### Task 9: 웹앱 동작 `docs/app.js`

**Files:**
- Create: `docs/app.js`

- [ ] **Step 1: 구현**

```js
// docs/app.js — 카메라 → 스캔 틀 → ONNX 추론 → 미션 판정 → 알람 해제
import { CLASSES } from './classes.js';
import { decode, nms, pickMain, scanCrop, Streak } from './postprocess.js';

// ── 설정 (src/evaluate.py와 같은 값) ──
const INPUT = 320, CONF = 0.5, IOU = 0.45, MIN_AREA = 0.15, SCAN_RATIO = 0.7;
const NEED_MS = 500, GRACE_MS = 250;
const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

const $ = id => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), octx = overlay.getContext('2d');
const work = Object.assign(document.createElement('canvas'), { width: INPUT, height: INPUT });
const wctx = work.getContext('2d', { willReadFrequently: true });

let session = null, audio = null, alarmTimer = null, beepTimer = null, wakeLock = null;
let mission = null, running = false, startedAt = 0, scanPhase = 0;
const streak = new Streak(NEED_MS, GRACE_MS);

// ── 화면 전환 ──
function show(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.toggle('active', s.id === id));
}

// ── 한국어 조사 ──
const hasBatchim = w => { const c = w.charCodeAt(w.length - 1) - 0xac00; return c >= 0 && c <= 11171 && c % 28 !== 0; };
const missionText = c => `${c.emoji} ${c.ko}${hasBatchim(c.ko) ? '을' : '를'} 비춰라`;
const notText = c => `그건 ${c.ko}${hasBatchim(c.ko) ? '이' : '가'} 아니잖아요!`;

// ── 모델 ──
async function loadModel() {
  if (session) return;
  $('status').textContent = '모델 불러오는 중…';
  ort.env.wasm.wasmPaths = ORT_DIST;
  session = await ort.InferenceSession.create('model.onnx', { executionProviders: ['wasm'] });
  $('status').textContent = '준비 완료';
}

// ── 소리 (WebAudio: 파일 없이 삐삐 소리) ──
function ensureAudio() {
  audio ??= new (window.AudioContext || window.webkitAudioContext)();
  return audio.resume();
}
function beep(freq, ms, when = 0) {
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = freq; o.type = 'square';
  g.gain.setValueAtTime(0.25, audio.currentTime + when);
  g.gain.setValueAtTime(0, audio.currentTime + when + ms / 1000);
  o.connect(g).connect(audio.destination);
  o.start(audio.currentTime + when); o.stop(audio.currentTime + when + ms / 1000 + 0.05);
}
const startAlarmSound = () => { beepTimer = setInterval(() => { beep(880, 150); beep(880, 150, 0.25); }, 1000); };
const stopAlarmSound = () => { clearInterval(beepTimer); beepTimer = null; };
const successSound = () => { beep(660, 120); beep(990, 200, 0.15); };

// ── 화면 켜짐 유지 ──
async function keepAwake() {
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch { /* 지원 안 하면 무시 */ }
}
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && alarmTimer) keepAwake(); });

// ── 카메라 ──
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  video.srcObject = stream;
  await video.play();
}
function stopCamera() { video.srcObject?.getTracks().forEach(t => t.stop()); video.srcObject = null; }

// ── 전처리: 가운데 스캔 틀 → 320×320 → [1,3,320,320] 0~1 ──
function toTensor() {
  const [sx, sy, s] = scanCrop(video.videoWidth, video.videoHeight, SCAN_RATIO);
  wctx.drawImage(video, sx, sy, s, s, 0, 0, INPUT, INPUT);
  const px = wctx.getImageData(0, 0, INPUT, INPUT).data, n = INPUT * INPUT, f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { f[i] = px[4 * i] / 255; f[n + i] = px[4 * i + 1] / 255; f[2 * n + i] = px[4 * i + 2] / 255; }
  return new ort.Tensor('float32', f, [1, 3, INPUT, INPUT]);
}

// ── 화면 좌표: video는 object-fit: cover로 가운데 정렬 → 스캔 틀도 화면 가운데 ──
function screenBox() {
  const dpr = window.devicePixelRatio || 1;
  const W = overlay.clientWidth * dpr, H = overlay.clientHeight * dpr;
  if (overlay.width !== W || overlay.height !== H) { overlay.width = W; overlay.height = H; }
  const k = Math.max(W / video.videoWidth, H / video.videoHeight);       // cover 배율
  const side = SCAN_RATIO * Math.min(video.videoWidth, video.videoHeight) * k;
  return { x: (W - side) / 2, y: (H - side) / 2, side, W, H, dpr };
}

function draw(dets, main, progress) {
  const b = screenBox(), c = octx, u = b.dpr;
  c.clearRect(0, 0, b.W, b.H);
  c.fillStyle = 'rgba(0,0,0,0.55)'; c.fillRect(0, 0, b.W, b.H);        // 바깥 어둡게
  c.clearRect(b.x, b.y, b.side, b.side);                                 // 틀 안은 원본
  // 모서리 틀
  const L = b.side * 0.12; c.strokeStyle = progress > 0 ? '#3ddc97' : '#fff'; c.lineWidth = 5 * u; c.lineCap = 'round';
  for (const [x, y, dx, dy] of [[b.x, b.y, 1, 1], [b.x + b.side, b.y, -1, 1], [b.x, b.y + b.side, 1, -1], [b.x + b.side, b.y + b.side, -1, -1]]) {
    c.beginPath(); c.moveTo(x + dx * L, y); c.lineTo(x, y); c.lineTo(x, y + dy * L); c.stroke();
  }
  // 움직이는 스캔 선
  scanPhase = (scanPhase + 0.02) % 2;
  const ly = b.y + b.side * (scanPhase < 1 ? scanPhase : 2 - scanPhase);
  c.strokeStyle = 'rgba(61,220,151,0.8)'; c.lineWidth = 2 * u; c.beginPath(); c.moveTo(b.x + 8 * u, ly); c.lineTo(b.x + b.side - 8 * u, ly); c.stroke();
  // 박스 (입력 320 좌표 → 화면 좌표)
  const s = b.side / INPUT;
  c.font = `${16 * u}px -apple-system, sans-serif`;
  for (const d of dets) {
    const hit = d === main && d.cls === mission.id;
    c.strokeStyle = hit ? '#3ddc97' : '#ffd166'; c.lineWidth = 3 * u;
    c.strokeRect(b.x + d.x1 * s, b.y + d.y1 * s, (d.x2 - d.x1) * s, (d.y2 - d.y1) * s);
    c.fillStyle = c.strokeStyle;
    c.fillText(`${CLASSES[d.cls].emoji} ${CLASSES[d.cls].ko} ${d.score.toFixed(2)}`, b.x + d.x1 * s + 4 * u, b.y + d.y1 * s - 6 * u);
  }
  // 진행 바
  const bw = b.side, bx = b.x, by = b.y + b.side + 24 * u;
  c.fillStyle = 'rgba(255,255,255,0.2)'; c.fillRect(bx, by, bw, 10 * u);
  c.fillStyle = '#3ddc97'; c.fillRect(bx, by, bw * progress, 10 * u);
}

// ── 메인 루프 ──
async function loop() {
  if (!running) return;
  if (video.readyState >= 2) {
    const out = await session.run({ [session.inputNames[0]]: toTensor() });
    const o = out[session.outputNames[0]], [, ch, n] = o.dims;
    const dets = nms(decode(o.data, ch - 4, n, CONF), IOU);
    const main = pickMain(dets, INPUT, MIN_AREA);
    const progress = streak.update(main !== null && main.cls === mission.id, performance.now());
    $('message').textContent = main && main.cls !== mission.id ? notText(CLASSES[main.cls]) : '';
    draw(dets, main, progress);
    if (progress >= 1) return success();
  }
  requestAnimationFrame(loop);
}

// ── 알람 울림 → 미션 ──
async function ring() {
  clearInterval(alarmTimer); alarmTimer = null;
  mission = CLASSES[Math.floor(Math.random() * CLASSES.length)];
  $('mission').textContent = missionText(mission);
  $('message').textContent = '';
  streak.reset();
  show('scan');
  startAlarmSound();
  await startCamera();
  startedAt = Date.now(); running = true;
  loop();
}

function success() {
  running = false;
  stopAlarmSound(); successSound();
  const snap = $('snapshot'), side = Math.min(video.videoWidth, video.videoHeight);
  snap.width = snap.height = side;
  const [sx, sy, s] = scanCrop(video.videoWidth, video.videoHeight, SCAN_RATIO);
  snap.getContext('2d').drawImage(video, sx, sy, s, s, 0, 0, side, side);
  stopCamera();
  wakeLock?.release(); wakeLock = null;
  const sec = Math.round((Date.now() - startedAt) / 1000);
  $('doneText').textContent = `${missionText(mission)} 미션을 ${sec}초 만에 통과했어요.`;
  show('done');
}

// ── 버튼 ──
$('arm').onclick = async () => {
  await ensureAudio(); await loadModel(); await keepAwake();
  // 카메라 권한을 미리 받아 둔다 (알람 시각에 권한 창이 뜨지 않게)
  await startCamera(); stopCamera();
  const [hh, mm] = $('time').value.split(':').map(Number);
  const target = new Date(); target.setHours(hh, mm, 0, 0);
  if (target <= new Date()) target.setDate(target.getDate() + 1);
  $('armedTime').textContent = $('time').value;
  show('armed');
  alarmTimer = setInterval(() => { if (new Date() >= target) ring(); }, 1000);
};
$('now').onclick = async () => { await ensureAudio(); await loadModel(); await keepAwake(); ring(); };
$('cancel').onclick = () => { clearInterval(alarmTimer); alarmTimer = null; wakeLock?.release(); wakeLock = null; show('setup'); };
$('again').onclick = () => show('setup');

loadModel().catch(e => { $('status').textContent = '모델 불러오기 실패: ' + e.message; });
```

- [ ] **Step 2: 맥 브라우저에서 확인**

```bash
cd docs && python3 -m http.server 8000
```
`http://localhost:8000` 을 Chrome으로 열고:
1. 상태가 "준비 완료"가 되는지
2. "지금 미션 시작" → 삐삐 소리, 카메라 화면, 가운데 틀·스캔 선·진행 바가 보이는지
3. 미션 물건을 틀에 대면 초록 박스 → 진행 바가 차고 → "🎉 기상 성공!" 화면
4. 다른 물건을 대면 "그건 ○○이 아니잖아요!"
5. 개발자 도구 콘솔에 에러가 없는지

- [ ] **Step 3: 커밋**

```bash
git add docs/app.js
git commit -m "feat: 웹앱 동작 (카메라 · 추론 · 미션 · 알람)"
```

---

### Task 10: 정식 학습 실행

- [ ] **Step 1: 노트북 정식 실행 (QUICK = False)**

```bash
.venv/bin/jupyter nbconvert --to notebook --execute wakeup_mission.ipynb --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=wakeup
```
Expected: 에러 없음. 데이터 받기 수십 분 + 학습 2회(맥 MPS 기준 각 30분~1시간 예상). 결과 표가 채워지고 `docs/model.onnx`(약 10MB)·`docs/classes.js` 생성.

- [ ] **Step 2: 결과 확인 · 커밋**

```bash
ls -la docs/model.onnx docs/classes.js
node --test tests/ && .venv/bin/python -m pytest tests -q
git add wakeup_mission.ipynb docs/model.onnx docs/classes.js
git commit -m "feat: 학습 결과 + ONNX 모델"
```

---

### Task 11: GitHub Pages 배포 (⚠️ 사용자 확인 후)

공개 저장소가 되면 노트북·SPEC·웹앱이 모두 공개된다. **실행 전에 사용자에게 저장소 이름과 공개 여부를 확인한다.**

- [ ] **Step 1: 저장소 만들고 올리기**

```bash
gh repo create wakeup-mission --public --source=. --push
gh api -X POST repos/{owner}/wakeup-mission/pages -f "source[branch]=main" -f "source[path]=/docs"
gh api repos/{owner}/wakeup-mission/pages --jq .html_url
```
Expected: `https://<계정>.github.io/wakeup-mission/`

- [ ] **Step 2: 아이폰에서 확인**

1~2분 뒤 아이폰 Safari로 주소를 열고 Task 9 Step 2의 1~4를 반복한다. 확인할 것:
- 후면 카메라가 열리는지 (HTTPS)
- 소리가 나는지 (무음 스위치가 켜져 있으면 WebAudio 소리가 안 날 수 있음 → 무음 해제)
- 초당 몇 번 추론하는지 체감 (느리면 `SCAN_RATIO`는 그대로 두고 모델 입력을 256으로 다시 내보내는 것을 고려)

---

## 완료 기준 (SPEC §11과 대응)

| SPEC 기준 | 확인하는 곳 |
|---|---|
| data.yaml 10종 + 클래스별 개수 표 | 노트북 §2, §3 |
| 기준선 · A · B 비교표 | 노트북 §7 |
| 클래스별 mAP50 · 혼동행렬 · 해석 | 노트북 §7 (해석 한 줄은 실행 후 사용자가 작성 또는 결과 보고 추가) |
| 직접 찍은 사진 미션 통과율 | 노트북 §9 |
| ONNX ↔ PyTorch 일치 | 노트북 §8 assert |
| 아이폰에서 알람 해제 | Task 11 Step 2 |
| 노트북 에러 없이 실행 | Task 7 Step 2 (QUICK), Task 10 Step 1 (정식) |
