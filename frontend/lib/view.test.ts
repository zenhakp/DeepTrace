// Run: npm test (Node's built-in test runner; no test library needed).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import type { Evidence, Job, Question } from "./types.ts";
import type { Answer } from "./types.ts";
import {
  answerLabel, canAsk, HISTORY_OPEN_BY_DEFAULT, historyToggleLabel, overlayPanel, probText, REGION_NOTE,
  scoreSentence, shouldPoll, stabilityBars, statusText, steps, summaryTiles, timeline, uploadPanel, verdict,
} from "./view.ts";

const EXAMPLES = new URL("../../docs/part2_evidence_examples/", import.meta.url);
const example = (name: string): Evidence =>
  JSON.parse(readFileSync(new URL(`example_${name}.json`, EXAMPLES), "utf-8"));

function job(evidence: Evidence | null, over: Partial<Job> = {}): Job {
  return {
    job_id: "0".repeat(32), file_name: "x.jpg", media_type: evidence?.media_type ?? "image",
    created_at: "", state: evidence ? "done" : "analyzing", explanation_state: "ready", error: null,
    prediction: evidence?.prediction ?? null, evidence, explanation: null, overlays: {}, questions: [], ...over,
  };
}

test("probabilities never round to certainty", () => {
  assert.equal(probText(0.6518), "0.65");
  assert.equal(probText(0.9994), "above 0.99");
  assert.equal(probText(0.003), "below 0.01");
  assert.equal(probText(1), "1.00");
  assert.equal(probText(0.5), "0.50");
});

test("NO_FACE never shows REAL or MANIPULATED", () => {
  const v = verdict(job(example("no_face")));
  assert.deepEqual(v, { kind: "no_face", headline: "No face found — no verdict" });
  assert.equal(scoreSentence(example("no_face")), null);
  // even if a broken document carried a label, NO_FACE wins
  const broken = { ...example("no_face"), prediction: "MANIPULATED" as const };
  assert.equal(verdict(job(broken)).kind, "no_face");
  assert.doesNotMatch(JSON.stringify(verdict(job(broken))), /REAL|MANIPULATED/);
});

test("an OK result shows the detector's label in the required wording", () => {
  const v = verdict(job(example("image")));
  assert.equal(v.kind, "label");
  assert.equal(v.headline, "Detector prediction: MANIPULATED");
  assert.equal(scoreSentence(example("image")), "The detector assigned a fake-class probability of 0.65.");
  assert.equal(scoreSentence(example("video")),
    "The detector assigned a mean fake-class probability of 0.99 over 11 frame(s) with a usable face.");
});

test("failed and running jobs have no verdict", () => {
  assert.equal(verdict(job(null, { state: "failed", error: { code: "analysis_failed", message: "x" } })).kind,
    "failed");
  assert.equal(verdict(job(null)).kind, "pending");
});

test("polling stops when nothing is running any more", () => {
  const ev = example("image");
  assert.equal(shouldPoll(job(null, { state: "queued" })), true);
  assert.equal(shouldPoll(job(ev, { explanation_state: "generating" })), true);
  for (const s of ["ready", "unavailable", "not_needed"] as const) {
    assert.equal(shouldPoll(job(ev, { explanation_state: s })), false);
  }
  assert.equal(shouldPoll(job(null, { state: "failed", explanation_state: "not_needed" })), false);
  assert.match(statusText(job(ev, { explanation_state: "pending" })), /Generating explanation/);
});

test("timeline maps time and probability into the box", () => {
  const ev = example("video");
  const g = timeline(ev.frames!, ev.suspicious_intervals, 404, 104);
  const pts = g.points.split(" ");
  assert.equal(pts.length, 11);
  assert.equal(pts[0].split(",")[0], "4");                 // t = 0 at the left padding
  assert.equal(g.threshold, 52);                           // p = 0.5 in the middle
  assert.deepEqual(g.intervals, [{ x: 4, width: 396 }]);   // 0.0-5.0 s spans the whole width
  assert.deepEqual(g.noFace, []);
  assert.deepEqual(g.ticks.map((t) => t.label), ["0 s", "1 s", "2 s", "3 s", "4 s", "5 s"]);
});

test("frames without a face are marked, not drawn as a score", () => {
  const frames = [
    { frame: 0, time_s: 0, status: "OK" as const, prediction: "REAL" as const, fake_probability: 0.1 },
    { frame: 12, time_s: 0.5, status: "NO_FACE" as const, prediction: null, fake_probability: null },
    { frame: 24, time_s: 1, status: "OK" as const, prediction: "REAL" as const, fake_probability: 0.2 },
  ];
  const g = timeline(frames, [], 104, 104);
  assert.equal(g.points.split(" ").length, 2);
  assert.deepEqual(g.noFace, [52]);
});

const question = (state: Question["state"]): Question =>
  ({ id: "q", question: "Why?", asked_at: "", state, answer: null, error: null });

test("polling continues while a question is pending or being answered", () => {
  const ev = example("image");
  assert.equal(shouldPoll(job(ev, { questions: [question("pending")] })), true);
  assert.equal(shouldPoll(job(ev, { questions: [question("answering")] })), true);
  assert.equal(shouldPoll(job(ev, { questions: [question("answered"), question("failed")] })), false);
});

test("NO_FACE: no regions section, the upload is labelled as such", () => {
  const p = overlayPanel(example("no_face"));
  assert.equal(p.showRegions, false);
  assert.equal(p.title, "Uploaded image (no face found)");
  assert.doesNotMatch(JSON.stringify(p), /suspicious region|MANIPULATED|REAL/);
  const ok = overlayPanel(example("image"));
  assert.equal(ok.showRegions, true);
  assert.equal(ok.caption, REGION_NOTE);
  assert.match(ok.caption ?? "", /not ground-truth manipulated pixels/);
  assert.match(overlayPanel(example("video")).title, /^Key-frame overlays/);
});

test("summary strip: NO_FACE carries no label and no score", () => {
  const tiles = summaryTiles(job(example("no_face")));
  assert.equal(tiles[0].value, "No face found");
  assert.ok(tiles.every((t) => t.tone === "neutral"));
  assert.doesNotMatch(JSON.stringify(tiles), /REAL|MANIPULATED|0\.\d/);
});

test("summary strip: OK result, amber tone only for MANIPULATED, never certainty", () => {
  const tiles = summaryTiles(job(example("video")));
  assert.deepEqual(tiles.map((t) => t.value), ["MANIPULATED", "0.99", "HIGH", "Video, 5.0 s"]);
  assert.equal(tiles[0].tone, "manipulated");
  assert.equal(tiles[1].label, "Mean fake-class probability");
  assert.equal(tiles[1].detail, "Uncalibrated detector output");
  assert.equal(tiles[3].detail, "11 of 11 sampled frames with a face");
  const real = { ...example("image"), prediction: "REAL" as const, fake_probability: 0.3 };
  assert.equal(summaryTiles(job(real))[0].tone, "real");
  assert.equal(summaryTiles(job(example("image")))[3].value, "Image, 612 x 408 px");
});

test("stability bars show each component once, with the driver marked", () => {
  const bars = stabilityBars(example("image"));
  assert.deepEqual(bars.map((b) => [b.name, b.value, b.driver]),
    [["prediction", 0.8333, false], ["confidence", 0.1797, true], ["localization", 0.8364, false]]);
  assert.deepEqual(stabilityBars(example("no_face")), []);
});

test("steps follow the job and the explanation", () => {
  assert.deepEqual(steps(job(null, { state: "analyzing", explanation_state: "waiting" })).map((s) => s.state),
    ["active", "waiting", "waiting"]);
  assert.deepEqual(steps(job(example("image"), { explanation_state: "generating" })).map((s) => s.state),
    ["done", "done", "active"]);
  assert.deepEqual(steps(job(example("no_face"), { explanation_state: "not_needed" })).map((s) => s.state),
    ["done", "done", "skipped"]);
  assert.deepEqual(steps(job(null, { state: "failed", explanation_state: "not_needed",
    error: { code: "analysis_failed", message: "" } })).map((s) => s.state), ["failed", "skipped", "skipped"]);
});

test("upload area: hidden once a file is chosen, both hidden while uploading", () => {
  assert.deepEqual(uploadPanel(false, false), { dropzone: true, fileCard: false, uploading: false });
  assert.deepEqual(uploadPanel(true, false), { dropzone: false, fileCard: true, uploading: false });
  assert.deepEqual(uploadPanel(true, true), { dropzone: false, fileCard: false, uploading: true });
  // Remove clears the file: the drop area comes back
  assert.equal(uploadPanel(false, false).dropzone, true);
});

test("previous analyses are hidden by default", () => {
  assert.equal(HISTORY_OPEN_BY_DEFAULT, false);
  assert.equal(historyToggleLabel(false, 5), "Show previous analyses (5)");
  assert.equal(historyToggleLabel(true, 5), "Hide previous analyses");
});

test("the question input is hidden while a question is queued or being answered", () => {
  assert.equal(canAsk([]), true);
  assert.equal(canAsk([question("pending")]), false);
  assert.equal(canAsk([question("answered"), question("answering")]), false);
  assert.equal(canAsk([question("answered")]), true);
  assert.equal(canAsk([question("failed")]), true);
});

test("answer labels: refusals read as outside this analysis, never as a fixed answer", () => {
  const a = (kind: Answer["kind"], source: Answer["source"] = "fixed"): Answer => ({
    question: "q", text: "t", source, sections: [], facts: [], notes: [], model: null, seconds: null, kind,
    suggestions: [] });
  assert.equal(answerLabel(a("not_covered")), "Outside this analysis");
  assert.equal(answerLabel(a("answer", "llm")), "Language model, checked against the evidence");
  assert.equal(answerLabel(a("correction")), "Corrected from the evidence");
  for (const k of ["answer", "correction", "not_covered", "no_face", "unavailable", "rejected"] as const) {
    assert.doesNotMatch(answerLabel(a(k)), /Fixed answer/);
  }
});
