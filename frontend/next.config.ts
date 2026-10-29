import type { NextConfig } from "next";

// The browser only talks to this Next.js server; /api/* is forwarded to the FastAPI backend, so both share one
// origin (no CORS setup needed) and the backend stays on 127.0.0.1.
const API = process.env.DEEPTRACE_API ?? "http://127.0.0.1:8000";

const config: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
  experimental: {
    // Next buffers forwarded request bodies up to 10 MB by default; uploads may be up to the backend's
    // app.max_upload_mb (100 MB), so allow a little more and let the backend refuse larger files (413).
    proxyClientMaxBodySize: "110mb",
  },
};

export default config;
