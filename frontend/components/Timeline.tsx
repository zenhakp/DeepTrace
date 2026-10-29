import { probText, timeline } from "@/lib/view.ts";
import type { Frame, Interval } from "@/lib/types.ts";

const W = 720;
const H = 200;
const AXIS = 22;   // room for the time labels
const LEFT = 30;   // room for the 0 / 0.5 / 1 labels

/** Fake-class probability per sampled frame (line and dots), the 0.5 decision line (dashed), suspicious
 *  intervals (amber band) and sampled frames without a face (grey ticks). Plain SVG: no chart library. */
export default function Timeline({ frames, intervals }: { frames: Frame[]; intervals: Interval[] }) {
  const plotH = H - AXIS;
  const g = timeline(frames, intervals, W - LEFT, plotH, 10);
  const shift = (x: number) => x + LEFT;
  return (
    <figure>
      <svg className="timeline" viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Fake-class probability per sampled frame over time">
        {[10, plotH / 2, plotH - 10].map((y) => <line key={y} className="grid" x1={LEFT} x2={W} y1={y} y2={y} />)}
        <text x={LEFT - 6} y={14} textAnchor="end">1</text>
        <text x={LEFT - 6} y={g.threshold + 4} textAnchor="end">0.5</text>
        <text x={LEFT - 6} y={plotH - 6} textAnchor="end">0</text>
        {g.intervals.map((i, k) => (
          <rect key={k} className="interval" x={shift(i.x)} y={10} width={i.width} height={plotH - 20} rx={3} />
        ))}
        <line className="threshold" x1={LEFT} x2={W} y1={g.threshold} y2={g.threshold} />
        {g.points && <polyline points={g.dots.map((d) => `${shift(d.x)},${d.y}`).join(" ")} />}
        {g.dots.map((d, k) => <circle key={k} cx={shift(d.x)} cy={d.y} r={3} />)}
        {g.noFace.map((x, k) => <line key={k} className="noface" x1={shift(x)} x2={shift(x)} y1={plotH - 16} y2={plotH - 4} />)}
        {g.ticks.map((t) => <text key={t.label} x={shift(t.x)} y={H - 4} textAnchor="middle">{t.label}</text>)}
      </svg>
      <div className="legend">
        <span className="l-line">Fake-class probability per sampled frame</span>
        <span className="l-threshold">0.5 decision line</span>
        <span className="l-interval">Suspicious interval</span>
        <span className="l-noface">Frame without a face</span>
      </div>
      <div style={{ marginTop: 12 }}>
        {intervals.length > 0 ? (
          <ul className="small">
            {intervals.map((i, k) => (
              <li key={k} className="num">
                {i.start_s.toFixed(1)} s to {i.end_s.toFixed(1)} s: {i.suspicious_frames} suspicious sampled frame(s),
                mean fake-class probability {probText(i.mean_fake_probability)}, maximum{" "}
                {probText(i.max_fake_probability)}.
              </li>
            ))}
          </ul>
        ) : <p className="small">No suspicious interval (2 or more sampled frames in a row leaning MANIPULATED) was found.</p>}
      </div>
    </figure>
  );
}
