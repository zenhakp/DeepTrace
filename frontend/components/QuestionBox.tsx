"use client";

import { useState } from "react";

import { ApiFailure, askQuestion } from "@/lib/api.ts";
import type { Question } from "@/lib/types.ts";
import { answerLabel, canAsk } from "@/lib/view.ts";

function Pending({ text }: { text: string }) {
  return <div className="pending"><span className="dots" aria-hidden="true"><i /><i /><i /></span>{text}</div>;
}

/** A short conversation about this file's evidence. Sending returns at once (the backend queues the question) and
 *  the results page polls until it is answered. One question at a time: the input is hidden until the open
 *  question is answered or has failed, so each answer can use the checked answers before it as context. */
export default function QuestionBox({ jobId, questions, onAsked }:
  { jobId: string; questions: Question[]; onAsked: () => void }) {
  const [question, setQuestion] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const open = canAsk(questions) && !sending;

  async function send(text: string) {
    setSending(true);
    setError(null);
    try {
      await askQuestion(jobId, text);
      setQuestion("");
      onAsked();
    } catch (err) {
      const f = err as ApiFailure;
      setError(`${f.message} (${f.code})`);
    } finally {
      setSending(false);
    }
  }

  const last = questions[questions.length - 1];
  return (
    <section className="card">
      <div className="card-head"><h2>Ask about this result</h2>
        <span className="muted small">Answers come from this file&apos;s analysis and how the method works</span></div>
      <div className="card-body">
        {questions.length === 0 && (
          <p className="muted small">Ask anything about this analysis, for example why the Evidence Stability has its
            band, what the regions mean, or how the robustness tests work.</p>
        )}
        <div className="thread">
          {questions.map((q) => (
            <div key={q.id} className="turn">
              <div className="bubble user"><p>{q.question}</p></div>
              <div className="bubble assistant">
                {q.state === "pending" && <Pending text="Queued. It is answered after any explanation in progress." />}
                {q.state === "answering" && <Pending text="Answering… usually under a minute on CPU." />}
                {q.state === "failed" && <p className="error">This question could not be answered ({q.error?.code}).
                  You can ask it again.</p>}
                {q.state === "answered" && q.answer && (
                  <>
                    <p>{q.answer.text}</p>
                    {q.answer.suggestions.length > 0 && (
                      <div className="suggestions">
                        {q.answer.suggestions.map((s) => (
                          <button key={s} type="button" className="chip" disabled={!open || q !== last}
                                  onClick={() => send(s)}>{s}</button>
                        ))}
                      </div>
                    )}
                    <div className="bubble-meta">
                      <span className={`badge ${q.answer.source === "llm" ? "llm" : ""}`}>{answerLabel(q.answer)}</span>
                      {q.answer.facts.length > 0 && (
                        <details><summary>Evidence used ({q.answer.sections.join(", ")})</summary>
                          <ul className="facts">{q.answer.facts.map((f, i) => <li key={i}>{f}</li>)}</ul>
                        </details>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
        {open && (
          <form className="qa-form" onSubmit={(e) => { e.preventDefault(); if (question.trim()) send(question); }}>
            <textarea rows={2} maxLength={500} value={question} onChange={(e) => setQuestion(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey && question.trim()) { e.preventDefault(); send(question); }
                      }}
                      placeholder={questions.length ? "Ask a follow-up question" :
                        "For example: Why is the Evidence Stability what it is?"} />
            <div className="qa-actions">
              <span className="muted small num">{question.length} / 500. Enter sends, Shift+Enter adds a line.</span>
              <button className="btn" type="submit" disabled={!question.trim()}>Ask</button>
            </div>
          </form>
        )}
        {sending && <Pending text="Sending…" />}
        {error && <p className="error">{error}</p>}
      </div>
    </section>
  );
}
