// docs/postprocess.js
// YOLO ONNX 출력 후처리. src/postprocess.py와 같은 동작.
// 출력 배열 모양: [1, 4+nc, n] → data[c * n + i] 가 i번째 위치의 c번째 값

export function decode(data, nc, n, conf, inputSize = 320) {
  const clip = v => Math.min(inputSize, Math.max(0, v));
  const dets = [];
  for (let i = 0; i < n; i++) {
    let cls = 0, score = -1;
    for (let c = 0; c < nc; c++) {
      const s = data[(4 + c) * n + i];
      if (s > score) { score = s; cls = c; }
    }
    if (score < conf) continue;
    const cx = data[i], cy = data[n + i], w = data[2 * n + i], h = data[3 * n + i];
    dets.push({ x1: clip(cx - w / 2), y1: clip(cy - h / 2), x2: clip(cx + w / 2), y2: clip(cy + h / 2), score, cls });
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

export function pickMain(dets, inputSize, minSide) {
  let best = null, bestArea = 0;
  for (const d of dets) {
    const area = (d.x2 - d.x1) * (d.y2 - d.y1) / (inputSize * inputSize);
    const longSide = Math.max(d.x2 - d.x1, d.y2 - d.y1) / inputSize;
    if (longSide >= minSide && area > bestArea) { best = d; bestArea = area; }
  }
  return best;
}

// 클래스마다 다른 신뢰도 기준: classes[cls].conf가 있으면 그 값, 없으면 defaultConf (src/missioneval.py apply_thr와 같음)
export const confOf = (classes, cls, defaultConf) => classes[cls]?.conf ?? defaultConf;
export function applyConf(dets, classes, defaultConf) {
  return dets.filter(d => d.score >= confOf(classes, d.cls, defaultConf));
}
// decode에 쓸 가장 낮은 기준 (이보다 낮은 박스는 어느 클래스에서도 통과 못 함)
export function minConf(classes, defaultConf) {
  return Math.min(...classes.map(c => c.conf ?? defaultConf));
}

export function scanCrop(w, h, ratio) {
  const s = Math.round(ratio * Math.min(w, h));
  return [Math.floor((w - s) / 2), Math.floor((h - s) / 2), s];
}

// 미션 물건이 needMs 동안 "연속으로" 보이면 진행률 1. graceMs 이하의 짧은 끊김(한두 프레임 놓침)은 봐준다.
// update의 graceMs를 주면 그 호출에서만 기본값 대신 쓴다 (느린 기기에서 프레임 간격에 맞춰 넓히기).
export class Streak {
  constructor(needMs, graceMs) { this.needMs = needMs; this.graceMs = graceMs; this.reset(); }
  reset() { this.start = null; this.lastSeen = null; }
  update(seen, now, graceMs = this.graceMs) {
    if (seen) {
      if (this.start === null) this.start = now;
      this.lastSeen = now;
    } else if (this.lastSeen === null || now - this.lastSeen > graceMs) {
      this.reset();
      return 0;
    }
    return Math.min(1, ((this.lastSeen ?? now) - this.start) / this.needMs);
  }
}
