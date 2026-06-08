/**
 * PoseGuide: a plain-language guide that explains what the Pose page does
 * and what each tool is for. Written for students with no prior knowledge.
 * Collapsible; remembers the open/closed choice in localStorage.
 */
import { useEffect, useState } from 'react'
import { BookOpen, X, ArrowRight } from 'lucide-react'

const LS_KEY = 'pose_guide_seen'

// ── COCO-17 keypoint layout for the reference diagram ─────────────────────────
// Positions are laid out on a simple standing figure (viewBox 0 0 200 330).
const KP: { i: number; name: string; x: number; y: number; side: 'L' | 'R' | 'C' }[] = [
  { i: 0,  name: 'nose',           x: 100, y: 32,  side: 'C' },
  { i: 1,  name: 'left eye',       x: 92,  y: 26,  side: 'L' },
  { i: 2,  name: 'right eye',      x: 108, y: 26,  side: 'R' },
  { i: 3,  name: 'left ear',       x: 84,  y: 30,  side: 'L' },
  { i: 4,  name: 'right ear',      x: 116, y: 30,  side: 'R' },
  { i: 5,  name: 'left shoulder',  x: 76,  y: 74,  side: 'L' },
  { i: 6,  name: 'right shoulder', x: 124, y: 74,  side: 'R' },
  { i: 7,  name: 'left elbow',     x: 60,  y: 122, side: 'L' },
  { i: 8,  name: 'right elbow',    x: 140, y: 122, side: 'R' },
  { i: 9,  name: 'left wrist',     x: 54,  y: 166, side: 'L' },
  { i: 10, name: 'right wrist',    x: 146, y: 166, side: 'R' },
  { i: 11, name: 'left hip',       x: 86,  y: 168, side: 'L' },
  { i: 12, name: 'right hip',      x: 114, y: 168, side: 'R' },
  { i: 13, name: 'left knee',      x: 83,  y: 232, side: 'L' },
  { i: 14, name: 'right knee',     x: 117, y: 232, side: 'R' },
  { i: 15, name: 'left ankle',     x: 81,  y: 294, side: 'L' },
  { i: 16, name: 'right ankle',    x: 119, y: 294, side: 'R' },
]
const BONES = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 6], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
]
const sideColor = (s: 'L' | 'R' | 'C') =>
  s === 'L' ? '#38bdf8' : s === 'R' ? '#fb923c' : '#e2e8f0'

// A real example photo (Unsplash CDN, stable). Falls back to hidden if it
// ever fails to load, so the layout never shows a broken image icon.
const EXAMPLE_PHOTO =
  'https://images.unsplash.com/photo-1518611012118-696072aa579a?w=640&q=80'

function KeypointDiagram() {
  const [photoOk, setPhotoOk] = useState(true)

  return (
    <div style={{
      background: 'var(--surface2)', border: '1px solid var(--border)',
      borderRadius: 12, padding: 18,
      display: 'flex', flexDirection: 'column', gap: 18,
    }}>

      {/* Row 1: example photo + what happens to it */}
      <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', alignItems: 'center' }}>
        {photoOk && (
          <figure style={{ margin: 0, flexShrink: 0, width: 230 }}>
            <img src={EXAMPLE_PHOTO} alt="People exercising"
              onError={() => setPhotoOk(false)}
              style={{
                width: '100%', height: 150, objectFit: 'cover',
                borderRadius: 10, display: 'block', border: '1px solid var(--border)',
              }} />
            <figcaption style={{ fontSize: 11, color: 'var(--text3)', marginTop: 6,
              textAlign: 'center' }}>
              A normal photo you might upload
            </figcaption>
          </figure>
        )}
        <div style={{ flex: 1, minWidth: 240 }}>
          <p style={{ fontSize: 13.5, color: 'var(--text2)', lineHeight: 1.7, margin: 0 }}>
            When you give the model a photo like this, it finds <Term>each person</Term> in
            the picture and places <Term>17 points</Term> on their body. The diagram below
            shows where those points go and the fixed number that each one always uses, so
            point <Mono>9</Mono> is always the left wrist on every person.
          </p>
        </div>
      </div>

      {/* Row 2: the labelled figure + the numbered list */}
      <div style={{ display: 'flex', gap: 28, flexWrap: 'wrap', alignItems: 'center',
        borderTop: '1px solid var(--border)', paddingTop: 18 }}>
        {/* Bigger figure */}
        <svg viewBox="0 0 200 330" style={{ width: 210, height: 346, flexShrink: 0 }}>
          {BONES.map(([a, b], i) => (
            <line key={i} x1={KP[a].x} y1={KP[a].y} x2={KP[b].x} y2={KP[b].y}
              stroke="var(--border2)" strokeWidth={3} strokeLinecap="round" />
          ))}
          {KP.map(k => (
            <g key={k.i}>
              <circle cx={k.x} cy={k.y} r={9} fill={sideColor(k.side)}
                stroke="#0d1117" strokeWidth={2} />
              <text x={k.x} y={k.y + 3.5} textAnchor="middle" fontSize={9.5}
                fontWeight={700} fill="#0d1117"
                fontFamily="JetBrains Mono, monospace">{k.i}</text>
            </g>
          ))}
        </svg>

        {/* Numbered name list */}
        <div style={{ flex: 1, minWidth: 240 }}>
          <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)',
            textTransform: 'uppercase', letterSpacing: '0.08em', margin: '0 0 12px' }}>
            The 17 points and their numbers
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '5px 18px' }}>
            {KP.map(k => (
              <div key={k.i} style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                <span style={{
                  width: 18, textAlign: 'right', fontSize: 12.5, fontWeight: 700,
                  color: sideColor(k.side), fontFamily: 'JetBrains Mono, monospace',
                }}>{k.i}</span>
                <span style={{ fontSize: 12.5, color: 'var(--text2)' }}>{k.name}</span>
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 18, marginTop: 16 }}>
            {[['#38bdf8', 'left side'], ['#fb923c', 'right side'], ['#e2e8f0', 'centre']].map(([c, l]) => (
              <span key={l} style={{ display: 'flex', alignItems: 'center', gap: 6,
                fontSize: 12, color: 'var(--text3)' }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: c }} />{l}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Small building blocks (kept local so styling stays consistent) ────────────
const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <p style={{
    fontSize: 11, fontWeight: 600, color: 'var(--text3)',
    textTransform: 'uppercase', letterSpacing: '0.08em',
    margin: '22px 0 10px',
  }}>{children}</p>
)

const Step = ({ n, children }: { n: number; children: React.ReactNode }) => (
  <div style={{ display: 'flex', gap: 11, marginBottom: 9 }}>
    <span style={{
      flexShrink: 0, width: 20, height: 20, borderRadius: '50%',
      background: 'var(--accent-s)', color: 'var(--accent)',
      fontSize: 11, fontWeight: 700, display: 'flex',
      alignItems: 'center', justifyContent: 'center',
      fontFamily: 'JetBrains Mono, monospace',
    }}>{n}</span>
    <p style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.55, margin: 0 }}>{children}</p>
  </div>
)

// One tool entry: a name plus a clear description, with a thin accent rule
const Tool = ({ name, children }: { name: string; children: React.ReactNode }) => (
  <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
    <div style={{ flexShrink: 0, width: 3, borderRadius: 2, background: 'var(--border2)' }} />
    <div>
      <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', margin: '0 0 2px' }}>{name}</p>
      <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.6, margin: 0 }}>{children}</p>
    </div>
  </div>
)

const Term = ({ children }: { children: React.ReactNode }) => (
  <span style={{ color: 'var(--text)', fontWeight: 500 }}>{children}</span>
)

const Mono = ({ children }: { children: React.ReactNode }) => (
  <span style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--accent)' }}>{children}</span>
)

// ── The full guide content (rendered inside the modal) ───────────────────────
function GuideContent() {
  return (
        <div style={{ padding: '4px 24px 24px' }}>

          {/* Intro */}
          <p style={{ fontSize: 13.5, color: 'var(--text2)', lineHeight: 1.7, margin: '16px 0 0' }}>
            <Term>Pose estimation</Term> is a model that looks at a picture of a person and
            marks <Term>17 points</Term> on their body, such as the nose, shoulders, elbows,
            wrists, hips, knees, and ankles. It then joins those points into a stick figure
            called a <Term>skeleton</Term>. This page lets you run that on a photo, a video,
            or your live webcam, and then study what the model produced.
          </p>

          <p style={{ fontSize: 13.5, color: 'var(--text2)', lineHeight: 1.7, margin: '12px 0 14px' }}>
            Every one of the 17 points carries three things. It has a <Term>name</Term> (for
            example &ldquo;left elbow&rdquo;), a <Term>position</Term> on the image, and a{' '}
            <Term>confidence</Term> from <Mono>0</Mono> to <Mono>1</Mono> that tells you how
            sure the model is about that point. Everything else on this page, including the
            angles, gestures, and rep counting, is built from these points.
          </p>

          {/* Keypoint diagram */}
          <KeypointDiagram />

          {/* Getting started */}
          <SectionLabel>Getting started</SectionLabel>
          <Step n={1}>
            Choose a mode at the top of the panel. <Term>Image</Term> lets you upload a photo,{' '}
            <Term>Video</Term> takes a file or a link, and <Term>Webcam</Term> uses your camera
            live.
          </Step>
          <Step n={2}>
            Run it. For a photo or webcam you will see the skeleton drawn over the person right
            away.
          </Step>
          <Step n={3}>
            Open the <Term>Learn &amp; Explore</Term> card on the left to turn the joint angles
            and the analysis panel on or off.
          </Step>
          <Step n={4}>
            Read the analysis below the image, then download the data if you want to use it
            somewhere else.
          </Step>

          {/* Tools */}
          <SectionLabel>What each tool does</SectionLabel>
          <Tool name="Skeleton">
            The stick figure drawn on the person. The colours tell you which side of the body a
            point belongs to. <span style={{ color: '#38bdf8' }}>Blue is the left side</span>,{' '}
            <span style={{ color: '#fb923c' }}>orange is the right side</span>, and white is the
            centre, meaning the nose and eyes.
          </Tool>
          <Tool name="Joint angles">
            The bend at a joint, measured in degrees. A straight arm is about <Mono>180°</Mono>,
            a right angle is <Mono>90°</Mono>, and a fully folded arm is close to <Mono>0°</Mono>.
            The number is drawn at each joint and also listed in the panel. This is how you check
            things like exercise form or posture.
          </Tool>
          <Tool name="Keypoint inspector">
            A table of all 17 points showing each name, its <Mono>x</Mono> and <Mono>y</Mono>{' '}
            position, and its confidence. This is the raw data the model gives you. Open it to
            see exactly what pose estimation produces underneath the skeleton.
          </Tool>
          <Tool name="Pose and gesture detection">
            Simple rules that read the points and name a pose, for example <Term>Hands Up</Term>,{' '}
            <Term>T&nbsp;Pose</Term>, or <Term>Standing</Term>. It shows how you can write your
            own rules on top of the points to recognise anything you want.
          </Tool>
          <Tool name="Exercise rep counter (webcam)">
            Counts repetitions of an exercise. Pick a <Term>Squat</Term>, a{' '}
            <Term>Push&nbsp;up</Term>, or a <Term>Bicep Curl</Term>. It watches one joint angle
            go down and then back up, and adds one to the count each time a full movement
            finishes.
          </Tool>
          <Tool name="Export to JSON or CSV">
            Download the points, angles, and gestures as a file. Use <Term>CSV</Term> to open
            the data in Excel or Google Sheets, or <Term>JSON</Term> to load it into a Python
            script or your own program.
          </Tool>

          {/* What you can learn */}
          <SectionLabel>What you can learn or build</SectionLabel>
          <ul style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 6 }}>
            {[
              'How a model turns a plain image into structured numbers you can work with.',
              'How to calculate an angle from three points, a core idea in geometry and biomechanics.',
              'How to write rules that recognise a pose or gesture from raw data.',
              'How to count exercise repetitions by tracking a joint angle over time.',
              'Real uses such as sports form analysis, physical therapy, fitness apps, and gesture controls.',
            ].map((t, i) => (
              <li key={i} style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.55 }}>{t}</li>
            ))}
          </ul>

          {/* Tips */}
          <SectionLabel>Tips for good results</SectionLabel>
          <div style={{
            background: 'var(--surface2)', border: '1px solid var(--border)',
            borderRadius: 8, padding: '12px 16px',
          }}>
            {[
              ['Stand fully in frame', 'The model needs to see the whole body to place all 17 points.'],
              ['View exercises from the side', 'Squats and pushups are easier to count when the camera sees you sideways.'],
              ['Use good, even lighting', 'Better light gives higher confidence and steadier points.'],
              ['Avoid baggy clothes', 'Loose clothing can hide where a joint really is.'],
            ].map(([t, d]) => (
              <div key={t} style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
                <span style={{ color: 'var(--success)', flexShrink: 0, fontSize: 13 }}>•</span>
                <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.55, margin: 0 }}>
                  <Term>{t}.</Term> {d}
                </p>
              </div>
            ))}
          </div>

          <p style={{ fontSize: 11.5, color: 'var(--text3)', lineHeight: 1.6, margin: '16px 0 0' }}>
            Note: this model finds where the body parts are, not who the person is. It does not
            recognise faces or identity.
          </p>
        </div>
  )
}

// ── Trigger banner + animated popup modal ────────────────────────────────────
export default function PoseGuide() {
  const [mounted, setMounted] = useState(false)   // is the modal in the DOM?
  const [closing, setClosing] = useState(false)   // playing the exit animation?

  const openGuide = () => { setClosing(false); setMounted(true) }
  const closeGuide = () => {
    setClosing(true)
    setTimeout(() => { setMounted(false); setClosing(false) }, 170)  // match guideCardOut
  }

  // Auto-open once for first-time visitors
  useEffect(() => {
    if (localStorage.getItem(LS_KEY) !== 'true') {
      localStorage.setItem(LS_KEY, 'true')
      const t = setTimeout(openGuide, 350)
      return () => clearTimeout(t)
    }
  }, [])

  // Escape to close + lock body scroll while open
  useEffect(() => {
    if (!mounted) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeGuide() }
    document.addEventListener('keydown', onKey)
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prevOverflow
    }
  }, [mounted])

  return (
    <>
      {/* Trigger banner */}
      <button onClick={openGuide} style={{
        width: '100%', display: 'flex', alignItems: 'center', gap: 11,
        padding: '13px 18px', marginBottom: 16, textAlign: 'left',
        background: 'var(--surface)', border: '1px solid var(--border)',
        borderRadius: 10, cursor: 'pointer', transition: 'border-color 0.15s, background 0.15s',
      }}
        onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent)' }}
        onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)' }}>
        <span style={{
          width: 32, height: 32, borderRadius: 8, flexShrink: 0,
          background: 'var(--accent-s)', display: 'flex',
          alignItems: 'center', justifyContent: 'center',
        }}>
          <BookOpen size={16} style={{ color: 'var(--accent)' }} />
        </span>
        <span style={{ flex: 1 }}>
          <span style={{ display: 'block', fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>
            What is this page, and what can you do here?
          </span>
          <span style={{ display: 'block', fontSize: 12, color: 'var(--text3)', marginTop: 1 }}>
            New to pose estimation? Open the quick guide.
          </span>
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0,
          fontSize: 12, fontWeight: 500, color: 'var(--accent)' }}>
          Open guide <ArrowRight size={14} />
        </span>
      </button>

      {/* Modal */}
      {mounted && (
        <div
          onClick={closeGuide}
          className={closing ? 'guide-backdrop-out' : 'guide-backdrop-in'}
          style={{
            position: 'fixed', inset: 0, zIndex: 200,
            background: 'rgba(8, 10, 14, 0.62)', backdropFilter: 'blur(3px)',
            display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
            padding: '6vh 16px 16px', overflowY: 'auto',
          }}>
          <div
            onClick={e => e.stopPropagation()}
            className={closing ? 'guide-card-out' : 'guide-card-in'}
            style={{
              width: '100%', maxWidth: 780, background: 'var(--surface)',
              border: '1px solid var(--border2)', borderRadius: 16,
              boxShadow: '0 24px 64px rgba(0,0,0,0.5)', overflow: 'hidden',
              maxHeight: '88vh', display: 'flex', flexDirection: 'column',
            }}>
            {/* Sticky header */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 11,
              padding: '16px 22px', borderBottom: '1px solid var(--border)',
              flexShrink: 0, background: 'var(--surface)',
            }}>
              <BookOpen size={17} style={{ color: 'var(--accent)', flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <p style={{ fontSize: 15, fontWeight: 600, color: 'var(--text)', margin: 0 }}>
                  Pose Estimation Guide
                </p>
                <p style={{ fontSize: 12, color: 'var(--text3)', margin: '1px 0 0' }}>
                  Everything on this page, explained simply.
                </p>
              </div>
              <button onClick={closeGuide} title="Close (Esc)" style={{
                width: 30, height: 30, borderRadius: 7, border: '1px solid var(--border)',
                background: 'var(--surface2)', color: 'var(--text3)', cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                transition: 'color 0.12s, border-color 0.12s',
              }}
                onMouseEnter={e => { e.currentTarget.style.color = 'var(--text)';
                  e.currentTarget.style.borderColor = 'var(--border2)' }}
                onMouseLeave={e => { e.currentTarget.style.color = 'var(--text3)';
                  e.currentTarget.style.borderColor = 'var(--border)' }}>
                <X size={16} />
              </button>
            </div>

            {/* Scrollable body */}
            <div style={{ overflowY: 'auto', flex: 1 }}>
              <GuideContent />
            </div>
          </div>
        </div>
      )}
    </>
  )
}
