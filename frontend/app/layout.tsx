import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "DeepTrace",
  description: "Deepfake detection with forensic evidence and an evidence-grounded explanation (local)",
};

/** The brand mark: a frame with a crosshair, drawn in SVG (no icon font or image file). */
function Mark() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
      <rect x="1.5" y="1.5" width="19" height="19" rx="4" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <circle cx="11" cy="11" r="4.2" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M11 3.5v3.3M11 15.2v3.3M3.5 11h3.3M15.2 11h3.3" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <div className="topbar-inner">
            <Link href="/" className="brand">
              <Mark />
              <span className="brand-name">DeepTrace</span>
              <span className="brand-tag">Forensic evidence viewer</span>
            </Link>
            <nav className="topbar-nav">
              <span className="muted">Runs locally</span>
              <Link href="/">New analysis</Link>
            </nav>
          </div>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
