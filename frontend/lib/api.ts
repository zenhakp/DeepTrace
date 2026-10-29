// Calls to the FastAPI backend through the Next.js /api proxy (next.config.ts).
import type { ApiError, Job, Question } from "./types.ts";

export class ApiFailure extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { cache: "no-store", ...init });
  } catch {
    throw new ApiFailure(0, "backend_unreachable", "The DeepTrace backend is not reachable. Is it running on port 8000?");
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const d = body?.detail as ApiError | string | undefined;
    if (res.status >= 500 && !d) {
      throw new ApiFailure(res.status, "backend_unreachable", "The DeepTrace backend is not reachable. Is it running on port 8000?");
    }
    throw new ApiFailure(res.status, typeof d === "object" ? d.code : `http_${res.status}`,
      typeof d === "object" ? d.message : String(d ?? res.statusText));
  }
  return body as T;
}

export interface Health {
  status: string;
  checkpoint_found: boolean;
  llm: { available: boolean; model: string; problem: string | null };
}

export const getHealth = () => call<Health>("/api/health");
export const listJobs = () => call<{ jobs: Job[] }>("/api/jobs");
export const getJob = (id: string) => call<Job>(`/api/jobs/${id}`);

export function uploadFile(file: File): Promise<Job> {
  const form = new FormData();
  form.append("file", file);
  return call<Job>("/api/jobs", { method: "POST", body: form });
}

export const askQuestion = (id: string, question: string) =>
  call<Question>(`/api/jobs/${id}/questions`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }),
  });
