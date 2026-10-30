"use client";

import { useRef } from "react";

/** A thumbnail that opens the full-size image in a native <dialog> (Esc, the Close button or a click outside
 *  closes it). No library needed. */
export default function Lightbox({ src, alt, caption }: { src: string; alt: string; caption?: string }) {
  const dialog = useRef<HTMLDialogElement>(null);
  return (
    <figure>
      <button type="button" className="thumb" onClick={() => dialog.current?.showModal()}
              aria-label={`Enlarge: ${alt}`}>
        {/* eslint-disable-next-line @next/next/no-img-element -- served by the backend, no optimisation needed */}
        <img src={src} alt={alt} />
      </button>
      {caption && <figcaption>{caption}</figcaption>}
      <dialog ref={dialog} className="lightbox" onClick={(e) => { if (e.target === dialog.current) dialog.current?.close(); }}>
        <div className="lightbox-bar">
          <span>{caption ?? alt}</span>
          <button type="button" onClick={() => dialog.current?.close()}>Close</button>
        </div>
        {/* eslint-disable-next-line @next/next/no-img-element -- full-size view of the same file */}
        <img src={src} alt={alt} />
      </dialog>
    </figure>
  );
}
