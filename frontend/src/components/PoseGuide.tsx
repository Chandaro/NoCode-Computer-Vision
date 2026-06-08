/**
 * PoseGuide — a plain-language guide that explains what the Pose page does
 * and what each tool is for. Written for students with no prior knowledge.
 * Collapsible; remembers the open/closed choice in localStorage.
 */
import { useState } from 'react'
import { BookOpen, ChevronDown, ChevronUp } from 'lucide-react'

const LS_KEY = 'pose_guide_open'

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

// One tool entry: a name + a clear description, with a thin accent rule
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

export default function PoseGuide() {
  const [open, setOpen] = useState(() => localStorage.getItem(LS_KEY) !== 'false')
  const toggle = () => {
    setOpen(o => { localStorage.setItem(LS_KEY, String(!o)); return !o })
  }

  return (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 10, overflow: 'hidden', marginBottom: 16,
    }}>
      {/* Header */}
      <button onClick={toggle} style={{
        width: '100%', display: 'flex', alignItems: 'center', gap: 10,
        padding: '13px 18px', background: 'transparent', border: 'none',
        cursor: 'pointer', textAlign: 'left',
      }}>
        <BookOpen size={15} style={{ color: 'var(--accent)', flexShrink: 0 }} />
        <span style={{ flex: 1 }}>
          <span style={{ display: 'block', fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>
            What is this page, and what can you do here?
          </span>
          <span style={{ display: 'block', fontSize: 12, color: 'var(--text3)', marginTop: 1 }}>
            A short guide — read this first if you are new to pose estimation
          </span>
        </span>
        {open ? <ChevronUp size={16} style={{ color: 'var(--text3)' }} />
              : <ChevronDown size={16} style={{ color: 'var(--text3)' }} />}
      </button>

      {open && (
        <div style={{ padding: '4px 18px 20px', borderTop: '1px solid var(--border)' }}>

          {/* Intro */}
          <p style={{ fontSize: 13.5, color: 'var(--text2)', lineHeight: 1.7, margin: '16px 0 0' }}>
            <Term>Pose estimation</Term> is a model that looks at a picture of a person and
            marks <Term>17 points</Term> on their body — the nose, eyes, shoulders, elbows,
            wrists, hips, knees, and ankles. It then joins those points into a stick figure,
            called a <Term>skeleton</Term>. This page lets you run that on a photo, a video,
            or your live webcam, and then study what the model produced.
          </p>

          <p style={{ fontSize: 13.5, color: 'var(--text2)', lineHeight: 1.7, margin: '12px 0 0' }}>
            Each of the 17 points has three things: a <Term>name</Term> (such as
            &ldquo;left elbow&rdquo;), a <Term>position</Term> on the image, and a{' '}
            <Term>confidence</Term> from <Mono>0</Mono> to <Mono>1</Mono> that says how sure
            the model is about that point. Everything else on this page — angles, gestures,
            rep counting — is built from these points.
          </p>

          {/* Getting started */}
          <SectionLabel>Getting started</SectionLabel>
          <Step n={1}>
            Choose a mode at the top of the panel: <Term>Image</Term> (upload a photo),{' '}
            <Term>Video</Term> (upload a file or paste a link), or <Term>Webcam</Term>{' '}
            (use your camera live).
          </Step>
          <Step n={2}>
            Run it. For a photo or webcam you will see the skeleton drawn over the person
            straight away.
          </Step>
          <Step n={3}>
            Open the <Term>Learn &amp; Explore</Term> card on the left to turn the joint
            angles and the analysis panel on or off.
          </Step>
          <Step n={4}>
            Read the analysis below the image, and download the data if you want to use it
            somewhere else.
          </Step>

          {/* Tools */}
          <SectionLabel>What each tool does</SectionLabel>
          <Tool name="Skeleton">
            The stick figure drawn on the person. The colours tell you which side of the body
            a point is on: <span style={{ color: '#38bdf8' }}>blue is the left side</span>,{' '}
            <span style={{ color: '#fb923c' }}>orange is the right side</span>, and white is
            the centre (nose, eyes).
          </Tool>
          <Tool name="Joint angles">
            The bend at a joint, measured in degrees. A straight arm is about <Mono>180°</Mono>,
            a right angle is <Mono>90°</Mono>, and a fully folded arm is near <Mono>0°</Mono>.
            The number is drawn at each joint and listed in the panel. This is how you check
            things like exercise form or posture.
          </Tool>
          <Tool name="Keypoint inspector">
            A table of all 17 points with their name, their <Mono>x</Mono> and <Mono>y</Mono>{' '}
            position, and their confidence. This is the raw data the model gives you — open it
            to see exactly what pose estimation produces underneath the skeleton.
          </Tool>
          <Tool name="Pose / gesture detection">
            Simple rules that read the points and name a pose — for example{' '}
            <Term>Hands Up</Term>, <Term>T-Pose</Term>, or <Term>Standing</Term>. It shows
            how you can write your own logic on top of the points to recognise anything you
            want.
          </Tool>
          <Tool name="Exercise rep counter (webcam)">
            Counts repetitions of an exercise. Pick <Term>Squat</Term>,{' '}
            <Term>Push-up</Term>, or <Term>Bicep Curl</Term>. It watches one joint angle go
            down and then back up, and adds one to the count each time a full movement is
            finished.
          </Tool>
          <Tool name="Export (JSON / CSV)">
            Download the points, angles, and gestures as a file. Use <Term>CSV</Term> to open
            the data in Excel or Google Sheets, or <Term>JSON</Term> to load it into a Python
            script or your own program.
          </Tool>

          {/* What you can learn */}
          <SectionLabel>What you can learn or build</SectionLabel>
          <ul style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 6 }}>
            {[
              'How a model turns a plain image into structured numbers you can work with.',
              'How to calculate an angle from three points — a core idea in geometry and biomechanics.',
              'How to write rules that recognise a pose or gesture from raw data.',
              'How to count exercise reps by tracking a joint angle over time.',
              'Real uses: sports form analysis, physical therapy, fitness apps, and gesture controls.',
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
              ['Use a side-on view for exercises', 'Squats and push-ups are easier to count from the side.'],
              ['Good, even lighting', 'Better light means higher confidence and steadier points.'],
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
            Note: this model finds <em>where</em> the body parts are, not <em>who</em> the
            person is. It does not recognise faces or identity.
          </p>
        </div>
      )}
    </div>
  )
}
