"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import Lightbox from "@/components/Lightbox.tsx";
import QuestionBox from "@/components/QuestionBox.tsx";
import Timeline from "@/components/Timeline.tsx";
import { ApiFailure, getJob } from "@/lib/api.ts";
import type { Evidence, ExplainedSection, Job } from "@/lib/types.ts";
import {
  overlayPanel, probText, SCORE_NOTE, scoreSentence, shouldPoll, stabilityBars, statusText, steps, summaryTiles,
  verdict,
} from "@/lib/view.ts";

const POLL_MS = 2000;

function Card({ title, aside, children }: { title: string; aside?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="card">
      <div className="card-head"><h2>{title}</h2>{aside}</div>
      <div className="card-body">{children}</div>
    </section>
  );
}

function Steps({ job }: { job: Job }) {
  return (
    <div className="steps" aria-label="Progress">
      {steps(job).map((s) => (
        <div key={s.label} className={`step ${s.state}`}>
          <span className="step-mark" aria-hidden="true" />
          <div><div className="step-label">{s.label}</div><div className="step-detail">{s.detail}</div></div>
        </div>
      ))}
    </div>
  );
}

function Summary({ job }: { job: Job }) {
  return (
    <div className="summary">
      {summaryTiles(job).map((t) => (
        <div key={t.label} className={`card tile ${t.tone}`}>
          <div className="eyebrow">{t.label}</div>
          <div className="tile-value">{t.value}</div>
          {t.detail && <div className="tile-detail">{t.detail}</div>}
        </div>
      ))}
    </div>
  );
}

function Result({ job, ev }: { job: Job; ev: Evidence | null }) {
  const v = verdict(job);
  if (v.kind === "failed") {
    return <Card title="Analysis failed"><p className="error">{job.error?.code}: {job.error?.message}</p>
      <p className="small muted">No verdict exists for this upload. Check the file and upload it again.</p></Card>;
  }
  if (!ev) return null;
  if (v.kind === "no_face") {
    return <Card title="No face found"><p>No face was found, so the detector made no prediction and no forensic
      checks were run. There is no verdict for this {ev.media_type}.</p></Card>;
  }
  return (
    <Card title="Detector result">
      <p>{scoreSentence(ev)}</p>
      {ev.confidence !== null && (
        <p className="small muted num">Confidence (the probability of the predicted class) is {probText(ev.confidence)};
          it is uncalibrated and is not forensic certainty.</p>
      )}
    </Card>
  );
}

function Overlay({ job, ev, section }: { job: Job; ev: Evidence; section?: ExplainedSection }) {
  const panel = overlayPanel(ev);
  const entries = Object.entries(job.overlays);
  const time = (k: string) => ev.frames?.find((f) => String(f.frame) === k)?.time_s;
  const message = ev.localization?.message ?? ev.localization?.note;
  if (entries.length === 0 && !panel.showRegions) return null;
  return (
    <Card title={panel.title}>
      {entries.length === 1 && entries[0][0] === "image" ? (
        <Lightbox src={entries[0][1]} alt={panel.title}
                  caption={panel.showRegions ? "Click to enlarge" : "The uploaded image. Click to enlarge."} />
      ) : (
        <div className="frames">
          {entries.map(([k, url]) => (
            <Lightbox key={k} src={url} alt={`Key frame ${k}`}
                      caption={`Key frame at ${time(k)?.toFixed(1) ?? "?"} s`} />
          ))}
        </div>
      )}
      {panel.showRegions ? (
        <div className="stack" style={{ marginTop: 16 }}>
          {message && <p className="callout">{message}</p>}
          <p className="note">{panel.caption}</p>
          {section && (
            <ul className="facts">{section.facts.filter((f) => f !== message).map((f, i) => <li key={i}>{f}</li>)}</ul>
          )}
        </div>
      ) : <p className="small muted" style={{ marginTop: 12 }}>{panel.caption}</p>}
    </Card>
  );
}

function Stability({ ev }: { ev: Evidence }) {
  const es = ev.evidence_stability;
  if (!es) return null;
  return (
    <Card title="Evidence Stability" aside={<span className="band"><span className="band-name">{es.band}</span></span>}>
      <p>{es.summary}</p>
      {es.band_capped && es.cap_reason && <p className="callout">{es.cap_reason}</p>}
      <div className="bars" aria-label="Stability components, 0 to 1, higher is more stable">
        {stabilityBars(ev).map((b) => (
          <div key={b.name} className={`bar-row${b.driver ? " driver" : ""}`}>
            <span>{b.name}</span>
            <span className="bar" role="img" aria-label={`${b.name} ${b.value.toFixed(2)}`}>
              <span style={{ width: `${Math.round(b.value * 100)}%` }} /></span>
            <span className="bar-value">{b.value.toFixed(2)}</span>
          </div>
        ))}
      </div>
      <p className="small muted">Component values, 0 to 1, higher is more stable. Amber: the component that lost the
        most points.</p>
      <p className="note">{es.note}</p>
    </Card>
  );
}

function Robustness({ ev }: { ev: Evidence }) {
  if (!ev.robustness) return null;
  const obs = ev.robustness_observations;
  return (
    <Card title="Robustness observations" aside={<span className="muted small num">{obs.length} finding(s)</span>}>
      {obs.length > 0 ? (
        <ul className="small">{obs.map((o, i) => (
          <li key={i}>{o.time_s !== undefined && `Key frame at ${o.time_s.toFixed(1)} s: `}{o.message}</li>
        ))}</ul>
      ) : <p className="small">No prediction flips or lost faces were observed in these perturbation runs.</p>}
      <p className="note">{ev.robustness.note}</p>
    </Card>
  );
}

function Video({ ev }: { ev: Evidence }) {
  if (ev.media_type !== "video" || !ev.frames) return null;
  return (
    <Card title="Over time" aside={<span className="muted small num">{ev.frames.length} sampled frames</span>}>
      <Timeline frames={ev.frames} intervals={ev.suspicious_intervals} />
      {ev.isolated_suspicious_frames.length > 0 && (
        <p className="small">Isolated suspicious frame(s) at {ev.isolated_suspicious_frames
          .map((i) => `${i.start_s?.toFixed(1)} s`).join(", ")} (too short to form an interval).</p>
      )}
      {ev.observations.length > 0 && <ul className="small">{ev.observations.map((o, i) => <li key={i}>{o.message}</li>)}</ul>}
    </Card>
  );
}

function ExplanationView({ job }: { job: Job }) {
  const ex = job.explanation;
  if (!ex) return null;
  const waiting = job.explanation_state === "pending" || job.explanation_state === "generating";
  return (
    <section className="card">
      <div className="card-head"><h2>Explanation</h2>
        {ex.model && <span className="muted small">{ex.model}, local, {ex.seconds} s</span>}</div>
      {(waiting || ex.notes.length > 0) && (
        <div className="card-body stack" style={{ paddingBottom: 0 }}>
          {waiting && <div className="pending"><span className="dots" aria-hidden="true"><i /><i /><i /></span>
            Generating explanation… The evidence is shown as text until the language-model explanation is ready
            (usually 1 to 2 minutes on CPU).</div>}
          {ex.notes.map((n, i) => <p key={i} className="note">{n}</p>)}
        </div>
      )}
      {ex.sections.filter((s) => s.key !== "limitations").map((s) => (
        <div key={s.key} className="section-row">
          <div className="section-head">
            <h3>{s.title}</h3>
            <span className={`badge ${s.source === "llm" ? "llm" : ""}`}>
              {s.source === "llm" ? "Language model, checked against the evidence" : "Evidence text"}</span>
          </div>
          {s.source === "llm" ? (
            <>
              <p>{s.text}</p>
              <details><summary>Evidence for this section ({s.facts.length})</summary>
                <ul className="facts">{s.facts.map((f, i) => <li key={i}>{f}</li>)}</ul>
              </details>
            </>
          ) : <ul className="facts">{s.facts.map((f, i) => <li key={i}>{f}</li>)}</ul>}
        </div>
      ))}
    </section>
  );
}

function Limitations({ ev }: { ev: Evidence }) {
  return (
    <Card title="Limitations and warnings" aside={<span className="muted small">Always shown in full</span>}>
      {ev.warnings.map((w, i) => <p key={i} className="callout">Warning: {w.message}</p>)}
      <ul className="small">{ev.limitations.map((l, i) => <li key={i}>{l}</li>)}</ul>
    </Card>
  );
}

export default function JobPage() {
  const { id } = useParams<{ id: string }>();
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    getJob(id).then((j) => { setJob(j); setError(null); })
      .catch((e: ApiFailure) => setError(`${e.message} (${e.code})`));
  }, [id]);

  useEffect(load, [load]);
  useEffect(() => {
    if (!job || !shouldPoll(job)) return;
    const t = setTimeout(load, POLL_MS);
    return () => clearTimeout(t);
  }, [job, load]);

  if (error && !job) return <p className="error">{error} <Link href="/">Back to upload</Link></p>;
  if (!job) return <p className="muted">Loading…</p>;
  const ev = job.evidence;
  const evidenceSection = job.explanation?.sections.find((s) => s.key === "evidence");
  return (
    <>
      <div className="page-title">
        <div>
          <p className="eyebrow">Analysis</p>
          <h1>{job.file_name}</h1>
          <p className="small muted num">Uploaded {job.created_at.replace("T", " ").replace("+00:00", " UTC")}
            {" "}<span className="mono">{job.job_id.slice(0, 8)}</span></p>
        </div>
        {job.state === "done" && (
          <a className="btn secondary" href={`/api/jobs/${job.job_id}/report`} download={`${job.file_name}_report.md`}>
            Download report (Markdown)</a>
        )}
      </div>
      <Steps job={job} />
      {error && <p className="error">{error}</p>}
      <div className="stack">
        <Summary job={job} />
        <p className="note">{SCORE_NOTE} {statusText(job)}</p>
        <Result job={job} ev={ev} />
        {ev && (
          <>
            <div className="grid-main">
              <div className="stack">
                <Overlay job={job} ev={ev} section={evidenceSection} />
                <Video ev={ev} />
              </div>
              <div className="stack">
                <Stability ev={ev} />
                <Robustness ev={ev} />
              </div>
            </div>
            <ExplanationView job={job} />
            <Limitations ev={ev} />
            <QuestionBox jobId={job.job_id} questions={job.questions} onAsked={load} />
          </>
        )}
      </div>
    </>
  );
}
