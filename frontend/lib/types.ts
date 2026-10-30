// Shapes returned by the FastAPI backend (src/deeptrace/app/api.py). Evidence fields follow
// docs/part2_evidence_schema.md (schema 1.0); only the fields the UI reads are typed, extra fields are ignored.

export type Label = "REAL" | "MANIPULATED";
export type JobState = "queued" | "analyzing" | "done" | "failed";
export type ExplanationState = "waiting" | "pending" | "generating" | "ready" | "unavailable" | "not_needed";

export interface Message { kind?: string; code?: string; message: string; frame?: number; time_s?: number }

export interface Frame {
  frame: number;
  time_s: number;
  status: "OK" | "NO_FACE";
  prediction: Label | null;
  fake_probability: number | null;
}

export interface Interval {
  start_s: number;
  end_s: number;
  suspicious_frames: number;
  mean_fake_probability: number;
  max_fake_probability: number;
}

export interface Evidence {
  schema_version: string;
  media_type: "image" | "video";
  status: "OK" | "NO_FACE";
  input: { file_name: string; width: number; height: number; duration_s?: number };
  score_semantics: string;
  prediction: Label | null;
  confidence: number | null;
  fake_probability: number | null;
  face: { usable_face_frames?: number; sampled_frames?: number };
  localization: { message?: string; note?: string; evidence_strength?: string } | null;
  evidence_stability: {
    score: number;
    band: "HIGH" | "MEDIUM" | "LOW";
    band_capped: boolean;
    cap_reason: string | null;
    summary: string;
    note: string;
    components: { prediction: number; confidence: number; localization: number };
    points_lost: { prediction: number; confidence: number; localization: number };
    driver: string;
  } | null;
  frames: Frame[] | null;
  suspicious_intervals: Interval[];
  isolated_suspicious_frames: { start_s?: number }[];
  robustness: { note: string } | null;
  robustness_observations: Message[];
  observations: Message[];
  warnings: Message[];
  limitations: string[];
}

export interface ExplainedSection {
  key: string;
  title: string;
  facts: string[];
  text: string;
  source: "llm" | "template";
  problems: string[];
}

export interface Explanation {
  sections: ExplainedSection[];
  notes: string[];
  model: string | null;
  seconds: number | null;
}

export interface Answer {
  question: string;
  text: string;
  source: "llm" | "fixed";
  sections: string[];
  facts: string[];
  notes: string[];
  model: string | null;
  seconds: number | null;
  kind: "answer" | "correction" | "not_covered" | "no_face" | "unavailable" | "rejected";
  suggestions: string[];
}

export type QuestionState = "pending" | "answering" | "answered" | "failed";

export interface Question {
  id: string;
  question: string;
  asked_at: string;
  state: QuestionState;
  answer: Answer | null;
  error: { code: string; message: string } | null;
}

export interface Job {
  job_id: string;
  file_name: string;
  media_type: "image" | "video";
  created_at: string;
  state: JobState;
  explanation_state: ExplanationState;
  error: { code: string; message: string } | null;
  prediction: Label | null;
  evidence: Evidence | null;
  explanation: Explanation | null;
  overlays: Record<string, string>;
  questions: Question[];
}

export interface ApiError { code: string; message: string }
