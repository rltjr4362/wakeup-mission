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

let sessionPromise = null, session = null, audio = null, alarmTimer = null, beepTimer = null, wakeLock = null;
let mission = null, ringing = false, starting = false, running = false, startedAt = 0, scanPhase = 0;
let busy = false, avgFrameMs = 0;   // busy: 설정 버튼 처리 중 · avgFrameMs: 한 프레임 추론 시간(지수이동평균)
const streak = new Streak(NEED_MS, GRACE_MS);

// ── 화면 전환 ──
function show(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.toggle('active', s.id === id));
}

// ── 한국어 조사 ──
const hasBatchim = w => { const c = w.charCodeAt(w.length - 1) - 0xac00; return c >= 0 && c <= 11171 && c % 28 !== 0; };
const missionText = c => `${c.emoji} ${c.ko}${hasBatchim(c.ko) ? '을' : '를'} 비춰라`;
const notText = c => `그건 ${c.ko}${hasBatchim(c.ko) ? '이' : '가'} 아니잖아요!`;

// ── 모델 (여러 번 불려도 세션은 하나만 만든다) ──
function loadModel() {
  sessionPromise ??= (async () => {
    if (typeof ort === 'undefined') throw new Error('onnxruntime-web 스크립트를 불러오지 못했어요');
    $('status').textContent = '모델 불러오는 중…';
    ort.env.wasm.wasmPaths = ORT_DIST;
    session = await ort.InferenceSession.create('model.onnx', { executionProviders: ['wasm'] });
    $('status').textContent = '준비 완료';
    return session;
  })().catch(e => { sessionPromise = null; throw e; });   // 실패하면 다음에 다시 시도
  return sessionPromise;
}

// ── 소리 (WebAudio: 파일 없이 삐삐 소리). iOS는 사용자 터치 안에서 resume()해야 소리가 난다 ──
function ensureAudio() {
  audio ??= new (window.AudioContext || window.webkitAudioContext)();
  return audio.resume();
}
// Safari 16.4+: 'playback'이면 무음 스위치를 켜 둬도 WebAudio 소리가 난다
function setPlaybackSession() {
  try { if (navigator.audioSession) navigator.audioSession.type = 'playback'; } catch { /* 지원 안 하면 무시 */ }
}
function beep(freq, ms, when = 0) {
  if (!audio) return;
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = freq; o.type = 'square';
  g.gain.setValueAtTime(0.25, audio.currentTime + when);
  g.gain.setValueAtTime(0, audio.currentTime + when + ms / 1000);
  o.connect(g).connect(audio.destination);
  o.start(audio.currentTime + when); o.stop(audio.currentTime + when + ms / 1000 + 0.05);
}
const stopAlarmSound = () => { clearInterval(beepTimer); beepTimer = null; };
const startAlarmSound = () => {
  stopAlarmSound();
  beepTimer = setInterval(() => { audio?.resume(); beep(880, 150); beep(880, 150, 0.25); }, 1000);
};
const successSound = () => { beep(660, 120); beep(990, 200, 0.15); };

// ── 화면 켜짐 유지 (navigator.wakeLock이 없으면 무시) ──
async function keepAwake() {
  try { wakeLock = (await navigator.wakeLock?.request('screen')) ?? null; } catch { /* 지원 안 하면 무시 */ }
}
function releaseWake() { wakeLock?.release().catch(() => {}); wakeLock = null; }
// 화면이 꺼졌다 켜지면 wake lock이 풀리므로 다시 잡는다
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && (alarmTimer || ringing)) keepAwake();
});

// ── 카메라 (getUserMedia는 HTTPS 또는 localhost에서만 동작) ──
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('카메라를 쓸 수 없어요 (HTTPS 주소로 열어 주세요)');
  stopCamera();   // 이전 스트림이 남아 있으면 먼저 끈다 (카메라 누수 방지)
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  video.srcObject = stream;
  await video.play();
}
function stopCamera() { video.srcObject?.getTracks().forEach(t => t.stop()); video.srcObject = null; }

// ── 전처리: 가운데 스캔 틀 → 320×320 → [1,3,320,320] 0~1 ──
function toTensor() {
  const [sx, sy, s] = cropRect();
  wctx.drawImage(video, sx, sy, s, s, 0, 0, INPUT, INPUT);
  const px = wctx.getImageData(0, 0, INPUT, INPUT).data, n = INPUT * INPUT, f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { f[i] = px[4 * i] / 255; f[n + i] = px[4 * i + 1] / 255; f[2 * n + i] = px[4 * i + 2] / 255; }
  return new ort.Tensor('float32', f, [1, 3, INPUT, INPUT]);
}

// ── 스캔 틀 ──
// video는 object-fit: cover로 화면 가운데에 맞춰지고, 넘치는 부분은 잘려 보이지 않는다.
// 스캔 틀은 영상 전체가 아니라 화면에 "보이는" 영역의 짧은 변 × SCAN_RATIO(0.7)인 가운데 정사각형이다
// (가로 영상을 세로 화면에 띄워도 틀이 화면 안에 들어오게).
// cropRect: 영상 픽셀 좌표 [x, y, s]. screenBox: cover 배율 k = max(W/vw, H/vh)를 곱한 화면 좌표 (한 변 s·k, 화면 가운데).
function cropRect() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const k = Math.max(overlay.clientWidth / vw, overlay.clientHeight / vh);   // cover 배율
  const visible = Math.min(overlay.clientWidth, overlay.clientHeight) / k;  // 보이는 영역 짧은 변 (영상 픽셀)
  const [, , full] = scanCrop(vw, vh, 1);
  const s = Math.min(full, Math.round(SCAN_RATIO * visible));
  return [Math.floor((vw - s) / 2), Math.floor((vh - s) / 2), s];
}

function screenBox() {
  const dpr = window.devicePixelRatio || 1;
  const W = Math.round(overlay.clientWidth * dpr), H = Math.round(overlay.clientHeight * dpr);
  if (overlay.width !== W || overlay.height !== H) { overlay.width = W; overlay.height = H; }
  const k = Math.max(W / video.videoWidth, H / video.videoHeight);
  const side = cropRect()[2] * k;
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
    const cls = CLASSES[d.cls];
    c.fillText(`${cls ? cls.emoji + ' ' + cls.ko : d.cls} ${d.score.toFixed(2)}`, b.x + d.x1 * s + 4 * u, b.y + d.y1 * s - 6 * u);
  }
  // 진행 바
  const bw = b.side, bx = b.x, by = b.y + b.side + 24 * u;
  c.fillStyle = 'rgba(255,255,255,0.2)'; c.fillRect(bx, by, bw, 10 * u);
  c.fillStyle = '#3ddc97'; c.fillRect(bx, by, bw * progress, 10 * u);
}

// ── 메인 루프 (추론을 await하므로 한 번에 한 프레임만 돈다) ──
async function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.videoWidth > 0) {
    try {
      const t0 = performance.now();
      const out = await session.run({ [session.inputNames[0]]: toTensor() });
      if (!running) return;
      const dt = performance.now() - t0;
      avgFrameMs = avgFrameMs ? 0.8 * avgFrameMs + 0.2 * dt : dt;
      const o = out[session.outputNames[0]], [, ch, n] = o.dims;     // [1, 4+nc, n]
      const dets = nms(decode(o.data, ch - 4, n, CONF, INPUT), IOU);
      const main = pickMain(dets, INPUT, MIN_AREA);
      // 느린 기기에서도 한 프레임은 놓쳐도 되게: grace ≥ 2.5 × 프레임 시간
      const grace = Math.max(GRACE_MS, 2.5 * avgFrameMs);
      const progress = streak.update(main !== null && main.cls === mission.id, performance.now(), grace);
      $('message').textContent = main && main.cls !== mission.id && CLASSES[main.cls] ? notText(CLASSES[main.cls]) : '';
      draw(dets, main, progress);
      if (progress >= 1) return success();
    } catch (e) {
      console.error(e);
      $('message').textContent = '인식 오류: ' + e.message;
    }
  }
  requestAnimationFrame(loop);
}

// ── 알람 울림 → 미션 (중복 호출 방지) ──
async function ring() {
  clearInterval(alarmTimer); alarmTimer = null;
  if (ringing) return;
  ringing = true;
  mission = CLASSES[Math.floor(Math.random() * CLASSES.length)];
  $('mission').textContent = missionText(mission);
  $('message').textContent = '';
  streak.reset();
  show('scan');
  startAlarmSound();
  startedAt = Date.now();
  await startScan();
}

async function startScan() {
  if (starting || running) return;
  starting = true;
  try {
    await loadModel();
    await startCamera();
  } catch (e) {
    $('message').textContent = `시작 실패: ${e.message} (화면을 눌러 다시 시도)`;   // 알람은 계속 울린다
    return;
  } finally { starting = false; }
  running = true;
  loop();
}
// 스캔 화면을 누를 때마다 소리를 깨운다 (iOS는 터치 안에서만 resume 가능).
// 카메라/모델 시작에 실패했으면 다시 시도한다.
$('scan').onclick = () => {
  ensureAudio().catch(() => {});
  if (!running && !starting && ringing) startScan();
};

function success() {
  running = false; ringing = false;
  stopAlarmSound(); successSound();
  // 스냅샷 = 모델이 본 스캔 틀 부분 그대로
  const snap = $('snapshot');
  const [sx, sy, s] = cropRect();
  snap.width = snap.height = s;
  snap.getContext('2d').drawImage(video, sx, sy, s, s, 0, 0, s, s);
  stopCamera();
  releaseWake();
  const sec = Math.round((Date.now() - startedAt) / 1000);
  $('doneText').textContent = `${missionText(mission)} 미션을 ${sec}초 만에 통과했어요.`;
  show('done');
}

// ── 버튼 ──
const fail = e => { releaseWake(); $('status').textContent = '오류: ' + e.message; };
// 처리 중에는 두 버튼을 모두 막는다 (두 번 누르기 · 둘 다 누르기 방지)
function setBusy(b) { busy = b; $('arm').disabled = b; $('now').disabled = b; }
// "HH:MM" → [시, 분]. 비었거나 이상하면 null
function parseTime(v) {
  const m = /^(\d{1,2}):(\d{2})$/.exec(v || '');
  if (!m) return null;
  const hh = Number(m[1]), mm = Number(m[2]);
  return hh < 24 && mm < 60 ? [hh, mm] : null;
}

$('arm').onclick = async () => {
  if (busy) return;
  const hm = parseTime($('time').value);
  if (!hm) { $('status').textContent = '알람 시각을 먼저 정해 주세요'; return; }
  setBusy(true);
  try {
    // resume()과 wake lock 요청은 await 전에, 즉 사용자 터치 안에서 시작한다
    setPlaybackSession();
    const resumed = ensureAudio(), awake = keepAwake();
    await resumed; await awake;
    await loadModel();
    // 카메라 권한을 미리 받아 둔다 (알람 시각에 권한 창이 뜨지 않게)
    await startCamera(); stopCamera();
  } catch (e) { return fail(e); }
  finally { setBusy(false); }
  const [hh, mm] = hm;
  const target = new Date(); target.setHours(hh, mm, 0, 0);
  if (target <= new Date()) target.setDate(target.getDate() + 1);
  $('armedTime').textContent = $('time').value;
  show('armed');
  clearInterval(alarmTimer);
  alarmTimer = setInterval(() => { if (new Date() >= target) ring(); }, 1000);
};
$('now').onclick = async () => {
  if (busy) return;
  setBusy(true);
  try {
    setPlaybackSession();
    const resumed = ensureAudio(), awake = keepAwake();
    await resumed; await awake;
    await loadModel();
  } catch (e) { return fail(e); }
  finally { setBusy(false); }
  ring();
};
$('cancel').onclick = () => { clearInterval(alarmTimer); alarmTimer = null; releaseWake(); show('setup'); };
$('again').onclick = () => show('setup');

loadModel().catch(e => { $('status').textContent = '모델 불러오기 실패: ' + e.message; });
