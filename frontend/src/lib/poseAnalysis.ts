/**
 * poseAnalysis.ts — educational pose-estimation utilities.
 *
 * All functions are PURE and operate on COCO-17 keypoints (normalised 0..1).
 * Students can read this file to learn exactly how raw keypoints turn into
 * joint angles, gestures, and exercise rep counts — then reuse it elsewhere.
 *
 * COCO-17 keypoint order (index → name):
 *   0 nose            5 left_shoulder   10 right_wrist    15 left_ankle
 *   1 left_eye        6 right_shoulder  11 left_hip       16 right_ankle
 *   2 right_eye       7 left_elbow      12 right_hip
 *   3 left_ear        8 right_elbow     13 left_knee
 *   4 right_ear       9 left_wrist      14 right_knee
 */

export const COCO_KEYPOINTS = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
] as const

export interface Pt { x: number; y: number }

const CONF_MIN = 0.3
const isValid = (kp: number[] | undefined, conf: number | undefined) =>
  !!kp && (conf ?? 0) >= CONF_MIN && !(kp[0] === 0 && kp[1] === 0)

/**
 * Angle (degrees, 0–180) at vertex B formed by points A-B-C.
 * This is the core biomechanics primitive — e.g. elbow angle =
 * angle at the elbow between the shoulder and the wrist.
 */
export function angleAt(a: Pt, b: Pt, c: Pt): number {
  const v1 = { x: a.x - b.x, y: a.y - b.y }
  const v2 = { x: c.x - b.x, y: c.y - b.y }
  const m1 = Math.hypot(v1.x, v1.y)
  const m2 = Math.hypot(v2.x, v2.y)
  if (m1 === 0 || m2 === 0) return NaN
  let cos = (v1.x * v2.x + v1.y * v2.y) / (m1 * m2)
  cos = Math.max(-1, Math.min(1, cos))
  return (Math.acos(cos) * 180) / Math.PI
}

/** The 8 joints we measure: vertex index + the two limb endpoints. */
export const JOINT_DEFS = [
  { name: 'L Elbow',    a: 5,  b: 7,  c: 9  },
  { name: 'R Elbow',    a: 6,  b: 8,  c: 10 },
  { name: 'L Shoulder', a: 7,  b: 5,  c: 11 },
  { name: 'R Shoulder', a: 8,  b: 6,  c: 12 },
  { name: 'L Hip',      a: 5,  b: 11, c: 13 },
  { name: 'R Hip',      a: 6,  b: 12, c: 14 },
  { name: 'L Knee',     a: 11, b: 13, c: 15 },
  { name: 'R Knee',     a: 12, b: 14, c: 16 },
] as const

export interface JointAngle {
  name: string
  vertex: number      // keypoint index of the joint
  angle: number       // degrees, NaN if not computable
  valid: boolean
  vx: number; vy: number   // vertex position (normalised) for overlay labels
}

/** Compute all joint angles for one person's keypoints. */
export function computeAngles(kps: number[][], conf: number[]): JointAngle[] {
  return JOINT_DEFS.map(j => {
    const ka = kps[j.a], kb = kps[j.b], kc = kps[j.c]
    const ok = isValid(ka, conf[j.a]) && isValid(kb, conf[j.b]) && isValid(kc, conf[j.c])
    const angle = ok
      ? angleAt({ x: ka[0], y: ka[1] }, { x: kb[0], y: kb[1] }, { x: kc[0], y: kc[1] })
      : NaN
    return { name: j.name, vertex: j.b, angle, valid: ok && !isNaN(angle),
             vx: kb ? kb[0] : 0, vy: kb ? kb[1] : 0 }
  })
}

/**
 * Rule-based gesture / pose detection. Returns a list of recognised labels.
 * Pure geometry on keypoints — a great example of building meaning on top
 * of a model's raw output.  (Image y grows downward: smaller y = higher up.)
 */
export function detectGestures(kps: number[][], conf: number[]): string[] {
  const out: string[] = []
  const g = (i: number): Pt | null =>
    isValid(kps[i], conf[i]) ? { x: kps[i][0], y: kps[i][1] } : null

  const lSh = g(5), rSh = g(6), lWr = g(9), rWr = g(10)
  const lEl = g(7), rEl = g(8), lHip = g(11), rHip = g(12)
  const lKn = g(13), rKn = g(14), lAn = g(15), rAn = g(16)
  const nose = g(0)

  // Hands up — both wrists above their shoulders
  if (lWr && rWr && lSh && rSh && lWr.y < lSh.y && rWr.y < rSh.y)
    out.push('🙌 Hands Up')
  // One hand raised
  else if ((lWr && lSh && lWr.y < lSh.y) || (rWr && rSh && rWr.y < rSh.y))
    out.push('✋ Hand Raised')

  // T-Pose — arms extended horizontally (wrists near shoulder height, far out)
  if (lWr && rWr && lSh && rSh && lEl && rEl) {
    const yTol = 0.08
    const armSpan = Math.abs(lWr.x - rWr.x)
    const shoulderSpan = Math.abs(lSh.x - rSh.x)
    const level = Math.abs(lWr.y - lSh.y) < yTol && Math.abs(rWr.y - rSh.y) < yTol
    if (level && armSpan > shoulderSpan * 1.8) out.push('🧍 T-Pose')
  }

  // Sitting vs standing — knee angle (bent ⇒ sitting/squat)
  const kneeAngles: number[] = []
  if (lHip && lKn && lAn) kneeAngles.push(angleAt(lHip, lKn, lAn))
  if (rHip && rKn && rAn) kneeAngles.push(angleAt(rHip, rKn, rAn))
  if (kneeAngles.length) {
    const avg = kneeAngles.reduce((a, b) => a + b, 0) / kneeAngles.length
    if (avg < 120) out.push('🪑 Sitting / Crouched')
    else if (avg > 160) out.push('🧍 Standing')
  }

  // Arms crossed (wrists on opposite side of body centre)
  if (lWr && rWr && lSh && rSh) {
    const cx = (lSh.x + rSh.x) / 2
    if (lWr.x > cx && rWr.x < cx) out.push('🤝 Arms Crossed')
  }

  // Leaning — shoulders significantly off-vertical from hips
  if (lSh && rSh && lHip && rHip && nose) {
    const shMid = { x: (lSh.x + rSh.x) / 2, y: (lSh.y + rSh.y) / 2 }
    const hipMid = { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 }
    const dx = shMid.x - hipMid.x
    const dy = Math.abs(shMid.y - hipMid.y) || 1e-6
    if (Math.abs(dx / dy) > 0.4) out.push('↗ Leaning')
  }

  return out
}

// ─── Exercise rep counting ────────────────────────────────────────────────────
//
// A rep is a full down→up cycle of a tracked joint angle.  We use a simple
// state machine: the angle must drop below `downThr` (the "down" position),
// then rise back above `upThr` (the "up" position) to score one rep.

export interface ExerciseDef {
  id: string
  label: string
  joints: { a: number; b: number; c: number }[]   // angles to average
  downThr: number      // below this = "down" phase
  upThr: number        // above this = "up" phase
  goodDepth?: number   // stricter "down" target — reps deeper than this are clean
  hint: string
}

export const EXERCISES: ExerciseDef[] = [
  {
    id: 'squat', label: 'Squat',
    joints: [{ a: 11, b: 13, c: 15 }, { a: 12, b: 14, c: 16 }], // both knees
    downThr: 100, upThr: 160, goodDepth: 90,
    hint: 'Face the camera side-on. Bend knees below 100°, then stand to 160°+.',
  },
  {
    id: 'pushup', label: 'Push-up',
    joints: [{ a: 5, b: 7, c: 9 }, { a: 6, b: 8, c: 10 }],       // both elbows
    downThr: 90, upThr: 150, goodDepth: 80,
    hint: 'Side-on view. Lower until elbows bend past 90°, then push back up.',
  },
  {
    id: 'curl', label: 'Bicep Curl',
    joints: [{ a: 5, b: 7, c: 9 }, { a: 6, b: 8, c: 10 }],       // both elbows
    downThr: 50, upThr: 150, goodDepth: 45,   // curled (small) → extended (large)
    hint: 'Curl arms up (elbow < 50°) then lower fully (elbow > 150°).',
  },
  {
    id: 'lunge', label: 'Lunge',
    joints: [{ a: 11, b: 13, c: 15 }, { a: 12, b: 14, c: 16 }], // both knees
    downThr: 110, upThr: 160, goodDepth: 95,
    hint: 'Step forward and bend the front knee toward 90°, then drive back up.',
  },
  {
    id: 'shoulder_press', label: 'Shoulder Press',
    joints: [{ a: 5, b: 7, c: 9 }, { a: 6, b: 8, c: 10 }],       // both elbows
    downThr: 100, upThr: 160, goodDepth: 95,   // racked → pressed overhead
    hint: 'Start with elbows bent at shoulders, press straight up to full extension.',
  },
  {
    id: 'situp', label: 'Sit-up',
    joints: [{ a: 5, b: 11, c: 13 }, { a: 6, b: 12, c: 14 }],   // shoulder-hip-knee
    downThr: 90, upThr: 150, goodDepth: 80,    // crunched up → lying back
    hint: 'Lie side-on to the camera. Crunch up until the torso folds past 90°.',
  },
  {
    id: 'jumping_jack', label: 'Jumping Jack',
    joints: [{ a: 7, b: 5, c: 11 }, { a: 8, b: 6, c: 12 }],     // elbow-shoulder-hip
    downThr: 40, upThr: 140, goodDepth: 130,   // arms down → arms overhead
    hint: 'Face the camera. Raise both arms fully overhead, then back to your sides.',
  },
]

export type RepPhase = 'up' | 'down'

export interface RepState {
  count: number
  phase: RepPhase
  angle: number       // current tracked angle
  minAngle: number    // deepest angle reached during the current down phase
  goodReps: number    // reps that hit the exercise's goodDepth target
  lastRepGood: boolean
  feedback: string    // live coaching cue
}

export function initRep(): RepState {
  return { count: 0, phase: 'up', angle: NaN, minAngle: NaN,
           goodReps: 0, lastRepGood: false, feedback: '' }
}

/**
 * Advance the rep state machine by one frame.
 * Tracks range-of-motion (deepest angle) to judge rep quality and give live
 * form feedback. Returns a NEW state (pure) plus whether a rep just completed.
 */
export function stepRep(
  state: RepState, kps: number[][], conf: number[], ex: ExerciseDef,
): { state: RepState; repJustCounted: boolean } {
  // Average the tracked joint angles that are valid this frame
  const vals: number[] = []
  for (const j of ex.joints) {
    const ka = kps[j.a], kb = kps[j.b], kc = kps[j.c]
    if (isValid(ka, conf[j.a]) && isValid(kb, conf[j.b]) && isValid(kc, conf[j.c])) {
      const ang = angleAt({ x: ka[0], y: ka[1] }, { x: kb[0], y: kb[1] }, { x: kc[0], y: kc[1] })
      if (!isNaN(ang)) vals.push(ang)
    }
  }
  if (!vals.length) return { state: { ...state, feedback: 'Step into frame…' }, repJustCounted: false }

  const angle = vals.reduce((a, b) => a + b, 0) / vals.length
  const good  = ex.goodDepth ?? ex.downThr
  let { count, phase, minAngle, goodReps, lastRepGood } = state
  let counted = false

  if (phase === 'up') {
    if (angle < ex.downThr) { phase = 'down'; minAngle = angle }
  } else {
    minAngle = isNaN(minAngle) ? angle : Math.min(minAngle, angle)
    if (angle > ex.upThr) {
      phase = 'up'; count += 1; counted = true
      lastRepGood = minAngle <= good
      if (lastRepGood) goodReps += 1
    }
  }

  // Live coaching cue
  let feedback: string
  if (phase === 'down') {
    feedback = (isNaN(minAngle) ? angle : minAngle) <= good ? 'Good depth — drive up!' : 'Go deeper…'
  } else if (counted) {
    feedback = lastRepGood ? `Rep ${count} ✓ clean` : `Rep ${count} — go deeper next time`
  } else {
    feedback = 'Lower down for a rep'
  }

  return { state: { count, phase, angle, minAngle, goodReps, lastRepGood, feedback },
           repJustCounted: counted }
}

// ─── Export helpers ───────────────────────────────────────────────────────────

/** Build a JSON-friendly object of a person's keypoints + angles + gestures. */
export function buildExport(kps: number[][], conf: number[]) {
  return {
    format: 'COCO-17',
    keypoints: COCO_KEYPOINTS.map((name, i) => ({
      index: i, name,
      x: kps[i]?.[0] ?? null,
      y: kps[i]?.[1] ?? null,
      confidence: conf[i] ?? null,
    })),
    joint_angles: computeAngles(kps, conf)
      .filter(a => a.valid)
      .map(a => ({ joint: a.name, angle_deg: Math.round(a.angle * 10) / 10 })),
    gestures: detectGestures(kps, conf),
  }
}

/** CSV string of the 17 keypoints (index,name,x,y,confidence). */
export function keypointsToCSV(kps: number[][], conf: number[]): string {
  const rows = ['index,name,x,y,confidence']
  COCO_KEYPOINTS.forEach((name, i) => {
    rows.push(`${i},${name},${kps[i]?.[0] ?? ''},${kps[i]?.[1] ?? ''},${conf[i] ?? ''}`)
  })
  return rows.join('\n')
}
