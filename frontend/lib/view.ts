// Display rules for the results page, kept as plain functions so they are tested (lib/view.test.ts).
// Rules from docs/part2_evidence_schema.md: the detector is the source of truth; NO_FACE has no verdict;
// scores are not certainty ("The detector assigned a fake-class probability of 0.65"); regions are
// model-indicated, not ground-truth manipulated pixels.

import type { Answer, Evidence, Frame, Interval, Job, Question } from "./types.ts";

export const SCORE_NOTE =
  "Scores are uncalibrated detector outputs, not probabilities that the media is manipulated.";
export const REGION_NOTE =
  "Highlighted regions are model-indicated suspicious regions, not ground-truth manipulated pixels. " +
  "Localization accuracy is not yet measured.";

/** A detector probability as text. 0.9994 must not read as "1.00" (certainty), nor 0.003 as "0.00". */
export function probText(p: number): string {
  if (p > 0.99 && p < 1) return "above 0.99";
  if (p > 0 && p < 0.01) return "below 0.01";
  return p.toFixed(2);
}

export type Verdict =
  | { kind: "pending"; headline: string }
  | { kind: "failed"; headline: string }
  | { kind: "no_face"; headline: string }
  | { kind: "label"; headline: string; label: "REAL" | "MANIPULATED" };

function hasVerdict(ev: Evidence | null): ev is Evidence & { prediction: "REAL" | "MANIPULATED" } {
  return !!ev && ev.status === "OK" && ev.prediction !== null;
}

/** The headline. A label is returned only for an OK evidence document that has one: never for NO_FACE. */
export function verdict(job: Job): Verdict {
  if (job.state === "failed") return { kind: "failed", headline: "Analysis failed — no verdict" };
  const ev = job.evidence;
  if (!ev) return { kind: "pending", headline: "Analysing…" };
  if (!hasVerdict(ev)) return { kind: "no_face", headline: "No face found — no verdict" };
  return { kind: "label", headline: `Detector prediction: ${ev.prediction}`, label: ev.prediction };
}

/** The score sentence in the required wording, or null when there is no score (NO_FACE). */
export function scoreSentence(ev: Evidence): string | null {
  if (!hasVerdict(ev) || ev.fake_probability === null) return null;
  const p = probText(ev.fake_probability);
  if (ev.media_type === "video") {
    const n = ev.face.usable_face_frames ?? 0;
    return `The detector assigned a mean fake-class probability of ${p} over ${n} frame(s) with a usable face.`;
  }
  return `The detector assigned a fake-class probability of ${p}.`;
}

export type Tone = "manipulated" | "real" | "neutral";
export interface Tile { label: string; value: string; detail?: string; tone: Tone }

/** The summary strip. NO_FACE and failed jobs carry no label and no score. */
export function summaryTiles(job: Job): Tile[] {
  const ev = job.evidence;
  const v = verdict(job);
  const media = !ev ? job.media_type
    : ev.media_type === "video"
      ? `Video, ${(ev.input.duration_s ?? 0).toFixed(1)} s`
      : `Image, ${ev.input.width} x ${ev.input.height} px`;
  const mediaDetail = ev?.media_type === "video" && ev.face.sampled_frames !== undefined
    ? `${ev.face.usable_face_frames ?? 0} of ${ev.face.sampled_frames} sampled frames with a face` : undefined;
  if (v.kind !== "label" || !ev) {
    const value = v.kind === "no_face" ? "No face found" : v.kind === "failed" ? "Analysis failed" : "Analysing";
    return [
      { label: "Detector result", value, detail: "No verdict", tone: "neutral" },
      { label: "Fake-class probability", value: "Not available", tone: "neutral" },
      { label: "Evidence Stability", value: "Not available", tone: "neutral" },
      { label: "Media", value: media, detail: mediaDetail, tone: "neutral" },
    ];
  }
  const es = ev.evidence_stability;
  return [
    { label: "Detector prediction", value: v.label, tone: v.label === "MANIPULATED" ? "manipulated" : "real" },
    { label: ev.media_type === "video" ? "Mean fake-class probability" : "Fake-class probability",
      value: probText(ev.fake_probability as number), detail: "Uncalibrated detector output", tone: "neutral" },
    { label: "Evidence Stability", value: es ? es.band : "Not available",
      detail: es ? `Score ${es.score.toFixed(2)}, project-defined` : undefined, tone: "neutral" },
    { label: "Media", value: media, detail: mediaDetail, tone: "neutral" },
  ];
}

export interface OverlayPanel { title: string; showRegions: boolean; caption: string | null }

/** NO_FACE: the uploaded image only, clearly labelled, and no regions section. */
export function overlayPanel(ev: Evidence): OverlayPanel {
  if (!hasVerdict(ev)) {
    return { title: ev.media_type === "video" ? "Uploaded video (no face found)" : "Uploaded image (no face found)",
             showRegions: false, caption: "No face was found, so no regions were computed." };
  }
  return { title: ev.media_type === "video" ? "Key-frame overlays and model-indicated suspicious regions"
                                            : "Overlay and model-indicated suspicious regions",
           showRegions: true, caption: REGION_NOTE };
}

export interface Bar { name: string; value: number; lost: number; driver: boolean }

/** The three Evidence Stability components as bars (0 to 1, higher is more stable). */
export function stabilityBars(ev: Evidence): Bar[] {
  const es = ev.evidence_stability;
  if (!es) return [];
  return (["prediction", "confidence", "localization"] as const).map((n) => ({
    name: n, value: es.components[n], lost: es.points_lost[n], driver: es.driver === n,
  }));
}

export type StepState = "done" | "active" | "waiting" | "skipped" | "failed";
export interface Step { label: string; state: StepState; detail: string }

export function steps(job: Job): Step[] {
  const a = job.state;
  const analysis: Step = a === "failed" ? { label: "Analysis", state: "failed", detail: job.error?.code ?? "failed" }
    : a === "done" ? { label: "Analysis", state: "done", detail: "Detector and forensic checks" }
    : { label: "Analysis", state: "active", detail: a === "queued" ? "Queued" : "Running on CPU" };
  const report: Step = a === "done" ? { label: "Evidence report", state: "done", detail: "From the evidence JSON" }
    : { label: "Evidence report", state: a === "failed" ? "skipped" : "waiting", detail: "After the analysis" };
  const e = job.explanation_state;
  const explanation: Step =
    e === "ready" ? { label: "Explanation", state: "done", detail: "Language model, checked" }
    : e === "pending" || e === "generating" ? { label: "Explanation", state: "active", detail: "Generating" }
    : e === "unavailable" ? { label: "Explanation", state: "skipped", detail: "Unavailable: evidence text shown" }
    : e === "not_needed" ? { label: "Explanation", state: "skipped", detail: "Not needed" }
    : { label: "Explanation", state: "waiting", detail: "After the report" };
  return [analysis, report, explanation];
}

/** Keep polling while the analysis, the LLM explanation or a question is still running. */
export function shouldPoll(job: Job): boolean {
  return job.state === "queued" || job.state === "analyzing" ||
    job.explanation_state === "pending" || job.explanation_state === "generating" ||
    job.questions.some((q) => q.state === "pending" || q.state === "answering");
}

export function statusText(job: Job): string {
  if (job.state === "queued") return "Queued.";
  if (job.state === "analyzing") return "Analysing on CPU, usually 15 to 60 seconds.";
  if (job.state === "failed") return `Analysis failed (${job.error?.code ?? "unknown error"}).`;
  switch (job.explanation_state) {
    case "pending":
    case "generating":
      return "Report ready. Generating explanation… usually 1 to 2 minutes.";
    case "ready":
      return "Report and explanation ready.";
    case "unavailable":
      return "Report ready. The language-model explanation is unavailable; the evidence is shown as text.";
    default:
      return "Report ready.";
  }
}

export interface TimelineGeometry {
  points: string;                                        // SVG polyline points for frames with a face
  dots: { x: number; y: number }[];                      // the same points, for markers
  threshold: number;                                     // y of fake-class probability 0.5
  intervals: { x: number; width: number }[];             // shaded suspicious intervals
  noFace: number[];                                      // x of sampled frames without a face
  ticks: { x: number; label: string }[];
}

/** Map frames (time_s, fake_probability) and intervals onto a width x height box (y = 1 at the top). */
export function timeline(frames: Frame[], intervals: Interval[], width: number, height: number,
                         pad = 4): TimelineGeometry {
  const end = Math.max(...frames.map((f) => f.time_s), ...intervals.map((i) => i.end_s), 0.001);
  const x = (t: number) => pad + (t / end) * (width - 2 * pad);
  const y = (p: number) => pad + (1 - p) * (height - 2 * pad);
  const r = (v: number) => Math.round(v * 10) / 10;
  const dots = frames.filter((f) => f.fake_probability !== null)
    .map((f) => ({ x: r(x(f.time_s)), y: r(y(f.fake_probability as number)) }));
  const step = end <= 10 ? 1 : end <= 60 ? 10 : 30;
  const ticks = [];
  for (let t = 0; t <= end + 1e-9; t += step) ticks.push({ x: r(x(t)), label: `${t} s` });
  return {
    points: dots.map((d) => `${d.x},${d.y}`).join(" "),
    dots,
    threshold: r(y(0.5)),
    intervals: intervals.map((i) => ({ x: r(x(i.start_s)), width: r(Math.max(x(i.end_s) - x(i.start_s), 2)) })),
    noFace: frames.filter((f) => f.fake_probability === null).map((f) => r(x(f.time_s))),
    ticks,
  };
}

// --- upload page ---

export interface UploadPanel { dropzone: boolean; fileCard: boolean; uploading: boolean }

/** The drop area shows only while nothing is chosen; the chosen file replaces it; while uploading, neither shows. */
export function uploadPanel(hasFile: boolean, busy: boolean): UploadPanel {
  if (busy) return { dropzone: false, fileCard: false, uploading: true };
  return { dropzone: !hasFile, fileCard: hasFile, uploading: false };
}

/** Previous analyses start collapsed on every visit (they stay on disk and their links keep working). */
export const HISTORY_OPEN_BY_DEFAULT = false;

export function historyToggleLabel(open: boolean, count: number): string {
  return open ? "Hide previous analyses" : `Show previous analyses (${count})`;
}

// --- questions ---

/** One question at a time: the input is hidden while a question is queued or being answered. */
export function canAsk(questions: Question[]): boolean {
  return !questions.some((q) => q.state === "pending" || q.state === "answering");
}

export function answerLabel(a: Answer): string {
  switch (a.kind) {
    case "answer": return a.source === "llm" ? "Language model, checked against the evidence" : "Checked answer";
    case "correction": return "Corrected from the evidence";
    case "not_covered": return "Outside this analysis";
    case "no_face": return "No verdict";
    case "unavailable": return "Language model unavailable";
    case "rejected": return "Answer withheld: failed the evidence checks";
    default: return "Outside this analysis";
  }
}
