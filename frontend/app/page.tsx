"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { ApiFailure, getHealth, listJobs, uploadFile, type Health } from "@/lib/api.ts";
import type { Job } from "@/lib/types.ts";
import { HISTORY_OPEN_BY_DEFAULT, historyToggleLabel, uploadPanel } from "@/lib/view.ts";

const ACCEPT = ".jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff,.mp4,.mov,.avi,.mkv,.webm,.m4v";

function jobLabel(j: Job): string {
  if (j.state === "failed") return `Failed (${j.error?.code})`;
  if (j.state !== "done") return j.state === "queued" ? "Queued" : "Analysing";
  return j.prediction ?? "No face found, no verdict";
}

function size(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function UploadGlyph() {
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true" className="muted">
      <rect x="3" y="7" width="28" height="22" rx="4" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M17 23V12M12.5 16.5 17 12l4.5 4.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

export default function UploadPage() {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[] | null>(null);       // loaded only when the list is opened
  const [historyOpen, setHistoryOpen] = useState(HISTORY_OPEN_BY_DEFAULT);
  const panel = uploadPanel(file !== null, busy);

  useEffect(() => {
    getHealth().then(setHealth).catch((e: ApiFailure) => setHealthError(e.message));
  }, []);

  function toggleHistory() {
    if (!historyOpen && jobs === null) listJobs().then((r) => setJobs(r.jobs)).catch(() => setJobs([]));
    setHistoryOpen(!historyOpen);
  }

  function choose(f: File | undefined) {
    setFile(f ?? null);
    setError(null);
  }

  async function analyse() {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const job = await uploadFile(file);
      router.push(`/jobs/${job.job_id}`);
    } catch (err) {
      const f = err as ApiFailure;
      setError(`${f.message} (${f.code})`);
      setBusy(false);
    }
  }

  return (
    <>
      <section className="hero">
        <p className="eyebrow">New analysis</p>
        <h1>Analyse an image or video</h1>
        <p>The detector analyses the largest face and is the source of truth. The forensic checks and the
          explanation only describe its evidence; scores are not probabilities that the media is manipulated.</p>
      </section>

      {panel.dropzone && (
      <div
        className={`dropzone${over ? " over" : ""}`} role="button" tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.current?.click(); } }}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); choose(e.dataTransfer.files[0]); }}
      >
        <UploadGlyph />
        <span className="dropzone-title">Drop a file here, or click to choose one</span>
        <span className="muted small">Images: jpg, png, bmp, webp, tif. Videos: mp4, mov, avi, mkv, webm, m4v.
          Up to 100 MB.</span>
        <input ref={input} type="file" accept={ACCEPT} hidden
               onChange={(e) => { choose(e.target.files?.[0]); e.target.value = ""; }} />
      </div>
      )}

      {panel.fileCard && file && (
        <div className="card file-card">
          <div className="file-card-info">
            <span className="eyebrow">Selected file</span>
            <strong className="file-card-name">{file.name}</strong>
            <span className="muted small num">{size(file.size)}</span>
          </div>
          <div className="file-card-actions">
            <button className="btn secondary" type="button" onClick={() => choose(undefined)}>Remove</button>
            <button className="btn" type="button" onClick={analyse}>Analyse</button>
          </div>
        </div>
      )}
      {panel.uploading && (
        <div className="card uploading">
          <span className="dots" aria-hidden="true"><i /><i /><i /></span>
          <span>Uploading {file?.name}… The analysis starts as soon as the upload is complete.</span>
        </div>
      )}
      {error && <p className="error" style={{ marginTop: 12 }}>{error}</p>}

      <div className="principles">
        <div className="card principle"><strong>Detector first</strong>
          <span>The prediction comes from the detector only; nothing else adds or changes a verdict.</span></div>
        <div className="card principle"><strong>Evidence you can check</strong>
          <span>Every explanation lists the evidence facts it is based on, and is checked against them.</span></div>
        <div className="card principle"><strong>Local and private</strong>
          <span>Files, analysis and the language model stay on this computer.</span></div>
      </div>

      {historyOpen && jobs !== null && (
        <section className="card">
          <div className="card-head"><h2>Previous analyses</h2><span className="muted small">{jobs.length}</span></div>
          {jobs.length === 0 && <p className="card-body muted small">No previous analyses.</p>}
          {jobs.length > 0 && <table>
            <thead><tr><th>File</th><th>Result</th><th>Uploaded (UTC)</th></tr></thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.job_id}>
                  <td><Link href={`/jobs/${j.job_id}`}>{j.file_name}</Link></td>
                  <td>{jobLabel(j)}</td>
                  <td className="muted">{j.created_at.replace("T", " ").replace("+00:00", "")}</td>
                </tr>
              ))}
            </tbody>
          </table>}
        </section>
      )}

      <div className="status-line">
        <button type="button" className="link-button" onClick={toggleHistory} aria-expanded={historyOpen}>
          {historyOpen ? historyToggleLabel(true, jobs?.length ?? 0) : "Show previous analyses"}</button>
        {healthError && <span className="warn">{healthError}</span>}
        {health && (
          <>
            <span className={health.checkpoint_found ? "ok" : "warn"}>
              {health.checkpoint_found ? "Detector checkpoint found" : "Detector checkpoint not found (set DEEPTRACE_ROOT)"}
            </span>
            <span className={health.llm.available ? "ok" : "warn"}>
              {health.llm.available ? `Explanation model ${health.llm.model} available`
                : "Explanation model not running: reports are shown as evidence text (start it with: ollama serve)"}
            </span>
          </>
        )}
      </div>
    </>
  );
}
