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
