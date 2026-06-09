/**
 * ConvBuilderGuide: a plain-language guide that explains what a CNN is and how
 * to build one on this page. Written for students with no prior knowledge.
 * Opens as an animated popup (same style as the Pose guide).
 */
import { useEffect, useState } from 'react'
import { BookOpen, X, ArrowRight } from 'lucide-react'

const LS_KEY = 'conv_guide_seen'

// Stable example photos (Unsplash CDN). onError hides them so a broken link
// never shows a broken-image icon.
const DOG_PHOTO = 'https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400&q=80'
const CAT_PHOTO = 'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400&q=80'

// ── CNN pipeline diagram (input image → conv → pool → flatten → output) ───────
function PipelineDiagram() {
  const [photoOk, setPhotoOk] = useState(true)
  const stages = [
    { label: 'Conv',    sub: 'find features', c: '#a78bfa' },
    { label: 'Pool',    sub: 'shrink',        c: '#38bdf8' },
    { label: 'Conv',    sub: 'find more',     c: '#a78bfa' },
    { label: 'Pool',    sub: 'shrink',        c: '#38bdf8' },
    { label: 'Flatten', sub: 'to a list',     c: '#fb923c' },
    { label: 'Output',  sub: 'cat or dog?',   c: '#3fb950' },
  ]
  return (
    <div style={{
      background: 'var(--surface2)', border: '1px solid var(--border)',
      borderRadius: 12, padding: 18,
    }}>
      <p style={{ fontSize: 13.5, color: 'var(--text2)', lineHeight: 1.7, margin: '0 0 14px' }}>
        A picture goes in on the left. Each layer changes it a little, pulling out more and
        more useful information, until the last layer makes a decision such as <strong>cat</strong>
        {' '}or <strong>dog</strong>.
      </p>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        {/* Input photo */}
        {photoOk && (
          <div style={{ textAlign: 'center', flexShrink: 0 }}>
            <img src={DOG_PHOTO} alt="input" onError={() => setPhotoOk(false)}
              style={{ width: 84, height: 84, objectFit: 'cover', borderRadius: 8,
                border: '1px solid var(--border2)', display: 'block' }} />
            <span style={{ fontSize: 10, color: 'var(--text3)' }}>input image</span>
          </div>
        )}
        <ArrowRight size={16} style={{ color: 'var(--text3)', flexShrink: 0 }} />

        {/* Stages */}
        {stages.map((s, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ textAlign: 'center', flexShrink: 0 }}>
              <div style={{
                width: 54, height: 54, borderRadius: 8,
                background: `${s.c}22`, border: `1.5px solid ${s.c}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 700, color: s.c,
              }}>{s.label}</div>
              <span style={{ fontSize: 9.5, color: 'var(--text3)', display: 'block', marginTop: 2 }}>{s.sub}</span>
            </div>
            {i < stages.length - 1 && <ArrowRight size={14} style={{ color: 'var(--text3)', flexShrink: 0 }} />}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Training figures (inline SVG, crisp + themed) ─────────────────────────────

// A loss curve dropping over epochs. Optional early-stop marker.
function LossCurveFig({ earlyStop = false }: { earlyStop?: boolean }) {
  return (
    <svg viewBox="0 0 230 110" style={{ width: '100%', maxWidth: 260, height: 'auto' }}>
      <line x1="26" y1="12" x2="26" y2="90" stroke="var(--border2)" strokeWidth="1.5" />
      <line x1="26" y1="90" x2="216" y2="90" stroke="var(--border2)" strokeWidth="1.5" />
      <text x="8" y="50" fontSize="9" fill="var(--text3)" transform="rotate(-90 8 50)">loss</text>
      <text x="120" y="105" fontSize="9" fill="var(--text3)" textAnchor="middle">epochs →</text>
      {/* high → drops fast → flattens */}
      <path d="M30 20 C 60 78, 95 84, 130 85 C 165 86, 195 86, 212 86"
        fill="none" stroke="var(--accent)" strokeWidth="2.5" />
      {earlyStop && (
        <>
          <line x1="130" y1="18" x2="130" y2="90" stroke="#f0a500" strokeWidth="1.5" strokeDasharray="4 3" />
          <circle cx="130" cy="85" r="4" fill="#f0a500" />
          <text x="134" y="32" fontSize="9" fill="#f0a500">stops here</text>
          <text x="134" y="43" fontSize="8" fill="var(--text3)">(no more improvement)</text>
        </>
      )}
    </svg>
  )
}

// One little "ball rolling into a valley" panel for learning rate.
function ValleyPanel({ kind, label }: { kind: 'low' | 'good' | 'high'; label: string }) {
  const dots =
    kind === 'low'  ? [[24, 30], [30, 36], [36, 41]]
    : kind === 'good' ? [[18, 26], [30, 48], [42, 62], [48, 67]]
    : [[22, 30], [66, 32], [28, 52], [62, 54], [30, 22]]   // bouncing
  const col = kind === 'good' ? '#3fb950' : kind === 'low' ? '#8b8b9a' : '#f87171'
  return (
    <div style={{ textAlign: 'center' }}>
      <svg viewBox="0 0 90 80" style={{ width: 90, height: 80 }}>
        <path d="M10 14 Q 45 82, 80 14" fill="none" stroke="var(--border2)" strokeWidth="2" />
        {kind === 'high' && (
          <path d="M30 22 L 24 10" stroke={col} strokeWidth="1.5" markerEnd="url(#arrow)" />
        )}
        <defs>
          <marker id="arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
            <path d="M0 0 L6 3 L0 6 Z" fill={col} />
          </marker>
        </defs>
        {dots.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="3.5" fill={col} opacity={0.45 + i * 0.14} />
        ))}
      </svg>
      <p style={{ fontSize: 10.5, color: col, fontWeight: 600, margin: 0 }}>{label}</p>
    </div>
  )
}

// Cosine vs Step learning-rate schedule curves.
function SchedulerFig() {
  const Curve = ({ d, title }: { d: string; title: string }) => (
    <div style={{ textAlign: 'center' }}>
      <svg viewBox="0 0 120 70" style={{ width: 120, height: 70 }}>
        <line x1="14" y1="8" x2="14" y2="58" stroke="var(--border2)" strokeWidth="1.2" />
        <line x1="14" y1="58" x2="112" y2="58" stroke="var(--border2)" strokeWidth="1.2" />
        <path d={d} fill="none" stroke="var(--accent)" strokeWidth="2" />
      </svg>
      <p style={{ fontSize: 10.5, color: 'var(--text3)', margin: 0 }}>{title}</p>
    </div>
  )
  return (
    <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
      {/* cosine: smooth half-wave down */}
      <Curve title="Cosine (smooth)" d="M18 14 C 50 16, 70 52, 108 54" />
      {/* step: staircase down */}
      <Curve title="Step (drops in stages)" d="M18 14 L 48 14 L 48 32 L 78 32 L 78 48 L 108 48" />
    </div>
  )
}

// Live augmentation strip: one real photo shown with CSS transforms.
function AugStrip() {
  const [ok, setOk] = useState(true)
  if (!ok) {
    // fallback if photo fails
    return null
  }
  const box: React.CSSProperties = {
    width: 78, height: 78, borderRadius: 8, overflow: 'hidden',
    border: '1px solid var(--border2)', flexShrink: 0,
  }
  const img: React.CSSProperties = { width: '100%', height: '100%', objectFit: 'cover', display: 'block' }
  const items: { label: string; t: string }[] = [
    { label: 'Original',  t: 'none' },
    { label: 'Flip LR',   t: 'scaleX(-1)' },
    { label: 'Flip UD',   t: 'scaleY(-1)' },
    { label: 'Rotation',  t: 'rotate(18deg) scale(1.25)' },
    { label: 'Translate', t: 'translateX(22%) scale(1.1)' },
    { label: 'Zoom',      t: 'scale(1.5)' },
  ]
  return (
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
      {items.map((it, i) => (
        <div key={i} style={{ textAlign: 'center' }}>
          <div style={box}>
            <img src={DOG_PHOTO} alt={it.label} onError={() => setOk(false)}
              style={{ ...img, transform: it.t }} />
          </div>
          <span style={{ fontSize: 10, color: 'var(--text3)' }}>{it.label}</span>
        </div>
      ))}
    </div>
  )
}

// ── Small building blocks ─────────────────────────────────────────────────────
const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <p style={{
    fontSize: 11, fontWeight: 600, color: 'var(--text3)',
    textTransform: 'uppercase', letterSpacing: '0.08em', margin: '22px 0 10px',
  }}>{children}</p>
)

const Step = ({ n, children }: { n: number; children: React.ReactNode }) => (
  <div style={{ display: 'flex', gap: 11, marginBottom: 9 }}>
    <span style={{
      flexShrink: 0, width: 20, height: 20, borderRadius: '50%',
      background: 'var(--accent-s)', color: 'var(--accent)', fontSize: 11, fontWeight: 700,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'JetBrains Mono, monospace',
    }}>{n}</span>
    <p style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.55, margin: 0 }}>{children}</p>
  </div>
)

// One layer entry: a coloured chip + name + clear description
const LayerRow = ({ name, color, children }:
  { name: string; color: string; children: React.ReactNode }) => (
  <div style={{ display: 'flex', gap: 11, marginBottom: 11 }}>
    <span style={{
      flexShrink: 0, marginTop: 1, padding: '2px 8px', borderRadius: 5, height: 'fit-content',
      fontSize: 10, fontWeight: 700, color, background: `${color}22`, border: `1px solid ${color}55`,
      fontFamily: 'JetBrains Mono, monospace', minWidth: 64, textAlign: 'center',
    }}>{name}</span>
    <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.6, margin: 0 }}>{children}</p>
  </div>
)

const Term = ({ children }: { children: React.ReactNode }) => (
  <span style={{ color: 'var(--text)', fontWeight: 500 }}>{children}</span>
)

// One training-setting entry: a name chip + description
const Setting = ({ name, children }: { name: string; children: React.ReactNode }) => (
  <div style={{ display: 'flex', gap: 11, marginBottom: 11 }}>
    <span style={{
      flexShrink: 0, marginTop: 1, padding: '2px 8px', borderRadius: 5, height: 'fit-content',
      fontSize: 10.5, fontWeight: 600, color: 'var(--text)', background: 'var(--surface3)',
      border: '1px solid var(--border)', minWidth: 86, textAlign: 'center',
    }}>{name}</span>
    <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.6, margin: 0 }}>{children}</p>
  </div>
)

// A figure with a caption, sits inside a setting block
const Figure = ({ children, caption }: { children: React.ReactNode; caption?: string }) => (
  <div style={{
    background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8,
    padding: '12px 14px', margin: '4px 0 14px',
  }}>
    {children}
    {caption && <p style={{ fontSize: 11, color: 'var(--text3)', margin: '8px 0 0', lineHeight: 1.5 }}>{caption}</p>}
  </div>
)

// ── Guide content ─────────────────────────────────────────────────────────────
function GuideContent() {
  const [catOk, setCatOk] = useState(true)
  return (
    <div style={{ padding: '4px 24px 24px' }}>

      {/* Intro */}
      <p style={{ fontSize: 13.5, color: 'var(--text2)', lineHeight: 1.7, margin: '16px 0 0' }}>
        A <Term>Convolutional Neural Network</Term> (CNN) is the kind of AI that looks at
        images and learns to tell them apart, for example a <Term>cat</Term> from a{' '}
        <Term>dog</Term>. On this page you build one yourself by stacking layers like building
        blocks, watch your image flow through it in 3D, and then train it on your own pictures.
        No code is needed.
      </p>

      <div style={{ display: 'flex', gap: 14, alignItems: 'center', margin: '14px 0' }}>
        {catOk && (
          <img src={CAT_PHOTO} alt="cat" onError={() => setCatOk(false)}
            style={{ width: 96, height: 96, objectFit: 'cover', borderRadius: 10,
              border: '1px solid var(--border2)', flexShrink: 0 }} />
        )}
        <p style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.7, margin: 0 }}>
          The network never sees the word &ldquo;cat&rdquo;. It only sees numbers (the pixel
          colours). Its job is to learn which patterns of numbers usually mean cat and which
          usually mean dog, all on its own, by looking at many labelled examples.
        </p>
      </div>

      {/* Pipeline */}
      <PipelineDiagram />

      {/* The layers */}
      <SectionLabel>The layers, and what each one does</SectionLabel>
      <LayerRow name="Conv 2D" color="#a78bfa">
        The most important layer. It slides small <Term>filters</Term> across the image to find
        patterns. Early conv layers find simple things like edges and colours. Later ones
        combine those into shapes, then whole objects like an ear or an eye.
      </LayerRow>
      <LayerRow name="ReLU" color="#fb923c">
        A simple rule that keeps the useful (positive) signals and sets everything else to
        zero. It lets the network learn complicated patterns instead of only straight lines.
      </LayerRow>
      <LayerRow name="Max Pool" color="#38bdf8">
        Shrinks the image by keeping only the strongest signal in each small area. This makes
        the network faster and helps it care about <Term>what</Term> is in the image rather
        than exactly <Term>where</Term>.
      </LayerRow>
      <LayerRow name="Batch Norm" color="#22d3ee">
        Keeps the numbers flowing through the network steady and balanced, which makes training
        faster and more reliable. A helper layer you usually place right after a Conv.
      </LayerRow>
      <LayerRow name="Dropout" color="#94a3b8">
        During training it randomly switches off some connections. This stops the network from
        simply memorising the training photos and helps it work on new, unseen images.
      </LayerRow>
      <LayerRow name="Flatten" color="#fb923c">
        Turns the 2D picture of features into one long list of numbers, so the final decision
        layers can read it.
      </LayerRow>
      <LayerRow name="Linear" color="#5e6ad2">
        Also called a dense layer. Every number connects to every output, which lets the
        network weigh up all the features and move toward a final answer.
      </LayerRow>
      <LayerRow name="Output" color="#3fb950">
        The last layer. It gives one score per class. The highest score is the network's
        guess, for example 0.92 cat and 0.08 dog.
      </LayerRow>

      {/* How to build */}
      <SectionLabel>How to build a network</SectionLabel>
      <Step n={1}>
        Start with a <Term>preset</Term> (Minimal, LeNet, VGG-mini) to get a working network,
        or build your own from the input down.
      </Step>
      <Step n={2}>
        Repeat the common pattern: <Term>Conv → ReLU → Max Pool</Term>. Each repeat lets the
        network find more detailed features while shrinking the image.
      </Step>
      <Step n={3}>
        Finish with <Term>Flatten → Linear → Output</Term>. The output classes are filled in
        automatically from your dataset.
      </Step>
      <Step n={4}>
        Watch the layer list. If a layer shows <Term>⚠ bad</Term>, the image has shrunk to
        nothing. Remove a pooling layer or make the input bigger.
      </Step>
      <Step n={5}>
        Set the training options on the right (epochs, batch, learning rate), then press{' '}
        <Term>Train</Term>.
      </Step>

      {/* Reading the 3D view */}
      <SectionLabel>Reading the 3D view</SectionLabel>
      <ul style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {[
          'Each block is one layer. The image flows from left (input) to right (output).',
          'Turn on Activations to see what each layer actually produces from a real image.',
          'Shuffle Image loads a different picture from your dataset so you can compare.',
          'Drag to orbit, scroll to zoom, or use Fly Mode to move through the network.',
          'The numbers under each block (like 16×64×64) are channels × height × width.',
        ].map((t, i) => (
          <li key={i} style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.55 }}>{t}</li>
        ))}
      </ul>

      {/* Tips */}
      <SectionLabel>Tips</SectionLabel>
      <div style={{
        background: 'var(--surface2)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '12px 16px',
      }}>
        {[
          ['Use the Conv, ReLU, Pool pattern', 'It is the proven building block of almost every image CNN.'],
          ['Do not stack too many pools', 'Each one halves the image. Too many and it disappears.'],
          ['More filters find more features', 'But they also use more memory and train slower.'],
          ['Bigger input keeps more detail', 'A 128×128 input sees more than 32×32, but costs more memory.'],
          ['Give each class enough images', 'Aim for at least 20 per class so the network has plenty to learn from.'],
        ].map(([t, d]) => (
          <div key={t} style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
            <span style={{ color: 'var(--success)', flexShrink: 0, fontSize: 13 }}>•</span>
            <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.55, margin: 0 }}>
              <Term>{t}.</Term> {d}
            </p>
          </div>
        ))}
      </div>

      {/* ════ TRAINING SETTINGS ════ */}
      <SectionLabel>Training settings on the right panel</SectionLabel>
      <p style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.7, margin: '0 0 14px' }}>
        Building the network decides its <Term>shape</Term>. These settings decide{' '}
        <Term>how it learns</Term>. Here is what each one means, with pictures.
      </p>

      {/* Basic */}
      <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent)', margin: '6px 0 8px' }}>BASIC</p>

      <Setting name="Epochs">
        One epoch means the network has looked at every training image once. More epochs means
        more practice. The loss (how wrong it is) drops quickly at first, then flattens out.
      </Setting>
      <Figure caption="Loss falls fast early on, then improvement slows. Too many epochs wastes time and can start memorising.">
        <LossCurveFig />
      </Figure>

      <Setting name="Batch">
        How many images the network looks at before it updates itself. Bigger batches are
        steadier but use more memory. 32 is a safe default.
      </Setting>

      <Setting name="LR">
        Learning rate: how big a step the network takes each update. This is the single most
        important setting.
      </Setting>
      <Figure caption="Think of rolling a ball into a valley. Too small = very slow. Too big = it overshoots and never settles. A good value reaches the bottom smoothly.">
        <div style={{ display: 'flex', gap: 22, flexWrap: 'wrap', justifyContent: 'center' }}>
          <ValleyPanel kind="low"  label="Too small" />
          <ValleyPanel kind="good" label="Just right" />
          <ValleyPanel kind="high" label="Too big" />
        </div>
      </Figure>

      <Setting name="Val Split %">
        Part of your images are held back and never trained on. They are used only to test how
        well the network does on pictures it has not seen. 20% is normal.
      </Setting>

      <Setting name="Patience">
        Early stopping. If the score does not improve for this many epochs in a row, training
        stops on its own so you do not waste time. 0 turns it off.
      </Setting>
      <Figure caption="Training stops automatically once the curve stops improving.">
        <LossCurveFig earlyStop />
      </Figure>

      {/* Optimizer */}
      <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent)', margin: '14px 0 8px' }}>OPTIMIZER</p>

      <Setting name="Algorithm">
        The method used to update the network. <Term>Adam</Term> is the easy, reliable default.
        <Term> SGD</Term> can reach slightly better results but needs more careful tuning.{' '}
        <Term>AdamW</Term> is Adam with cleaner weight decay.
      </Setting>
      <Setting name="Weight Decay">
        Gently pushes the network's numbers toward small values. This stops it from relying too
        heavily on any one feature and helps it work on new images. A tiny value like 0.0001 is
        common; 0 turns it off.
      </Setting>
      <Setting name="Momentum">
        Used by SGD. Like a rolling ball that keeps some speed from previous steps, so it pushes
        through small bumps instead of getting stuck. 0.9 is standard.
      </Setting>
      <Setting name="Warmup Ep">
        Start with a tiny learning rate for the first few epochs, then ramp up to the full
        value. This stops the network from taking a wild step at the very start.
      </Setting>

      {/* Scheduler */}
      <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent)', margin: '14px 0 8px' }}>LR SCHEDULER</p>
      <Setting name="Schedule">
        Slowly lowers the learning rate as training goes on, so the network takes big steps
        early and fine, careful steps near the end.
      </Setting>
      <Figure caption="Cosine lowers it smoothly. Step drops it in stages. Both help the network settle into a good answer.">
        <SchedulerFig />
      </Figure>

      {/* Regularisation */}
      <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent)', margin: '14px 0 8px' }}>REGULARISATION</p>
      <Setting name="Label Smooth">
        Instead of telling the network an answer is 100% certain, it says something like 95%.
        This keeps the network humble and often makes it more accurate on new images. 0.1 is a
        good value to try; 0 turns it off.
      </Setting>

      {/* Augmentation */}
      <p style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent)', margin: '14px 0 8px' }}>AUGMENTATION</p>
      <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.6, margin: '0 0 6px' }}>
        Augmentation makes free extra training examples by changing your photos slightly each
        time. A flipped or rotated dog is still a dog, so the network learns to recognise it
        from any angle and does not just memorise the exact pictures.
      </p>
      <Figure caption="The same photo, changed in different ways. Each value below sets how often or how strongly that change is applied.">
        <AugStrip />
      </Figure>
      <Setting name="Flip LR / UD">
        Mirror the image left to right, or top to bottom. The value is the chance it happens
        (0.5 means half the time). Left-right is usually safe; up-down only suits images that
        look fine upside down.
      </Setting>
      <Setting name="Rotation °">
        Turn the image by up to this many degrees. Helps with objects photographed at angles.
      </Setting>
      <Setting name="Translate">
        Shift the image sideways or up and down by a small amount, so the object is not always
        centred.
      </Setting>

      <p style={{ fontSize: 11.5, color: 'var(--text3)', lineHeight: 1.6, margin: '20px 0 0' }}>
        Note: this page uses the same images as the Image Classification page. Upload pictures
        for each class there first, then come back to build and train your CNN.
      </p>
    </div>
  )
}

// ── Trigger banner + animated popup modal ────────────────────────────────────
export default function ConvBuilderGuide() {
  const [mounted, setMounted] = useState(false)
  const [closing, setClosing] = useState(false)

  const open = () => { setClosing(false); setMounted(true) }
  const close = () => {
    setClosing(true)
    setTimeout(() => { setMounted(false); setClosing(false) }, 170)
  }

  useEffect(() => {
    if (localStorage.getItem(LS_KEY) !== 'true') {
      localStorage.setItem(LS_KEY, 'true')
      const t = setTimeout(open, 350)
      return () => clearTimeout(t)
    }
  }, [])

  useEffect(() => {
    if (!mounted) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.removeEventListener('keydown', onKey); document.body.style.overflow = prev }
  }, [mounted])

  return (
    <>
      {/* Trigger banner */}
      <button onClick={open} style={{
        width: '100%', display: 'flex', alignItems: 'center', gap: 11,
        padding: '13px 18px', marginBottom: 16, textAlign: 'left',
        background: 'var(--surface)', border: '1px solid var(--border)',
        borderRadius: 10, cursor: 'pointer', transition: 'border-color 0.15s',
      }}
        onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent)' }}
        onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)' }}>
        <span style={{
          width: 32, height: 32, borderRadius: 8, flexShrink: 0,
          background: 'var(--accent-s)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <BookOpen size={16} style={{ color: 'var(--accent)' }} />
        </span>
        <span style={{ flex: 1 }}>
          <span style={{ display: 'block', fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>
            What is a CNN, and how do I build one here?
          </span>
          <span style={{ display: 'block', fontSize: 12, color: 'var(--text3)', marginTop: 1 }}>
            New to neural networks? Open the quick guide.
          </span>
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0,
          fontSize: 12, fontWeight: 500, color: 'var(--accent)' }}>
          Open guide <ArrowRight size={14} />
        </span>
      </button>

      {/* Modal */}
      {mounted && (
        <div onClick={close}
          className={closing ? 'guide-backdrop-out' : 'guide-backdrop-in'}
          style={{
            position: 'fixed', inset: 0, zIndex: 200,
            background: 'rgba(8, 10, 14, 0.62)', backdropFilter: 'blur(3px)',
            display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
            padding: '6vh 16px 16px', overflowY: 'auto',
          }}>
          <div onClick={e => e.stopPropagation()}
            className={closing ? 'guide-card-out' : 'guide-card-in'}
            style={{
              width: '100%', maxWidth: 780, background: 'var(--surface)',
              border: '1px solid var(--border2)', borderRadius: 16,
              boxShadow: '0 24px 64px rgba(0,0,0,0.5)', overflow: 'hidden',
              maxHeight: '88vh', display: 'flex', flexDirection: 'column',
            }}>
            {/* Header */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 11,
              padding: '16px 22px', borderBottom: '1px solid var(--border)',
              flexShrink: 0, background: 'var(--surface)',
            }}>
              <BookOpen size={17} style={{ color: 'var(--accent)', flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <p style={{ fontSize: 15, fontWeight: 600, color: 'var(--text)', margin: 0 }}>
                  Conv Builder Guide
                </p>
                <p style={{ fontSize: 12, color: 'var(--text3)', margin: '1px 0 0' }}>
                  How CNNs work, and how to build one here, explained simply.
                </p>
              </div>
              <button onClick={close} title="Close (Esc)" style={{
                width: 30, height: 30, borderRadius: 7, border: '1px solid var(--border)',
                background: 'var(--surface2)', color: 'var(--text3)', cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}
                onMouseEnter={e => { e.currentTarget.style.color = 'var(--text)' }}
                onMouseLeave={e => { e.currentTarget.style.color = 'var(--text3)' }}>
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
