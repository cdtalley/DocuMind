import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const apiTarget = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8001").replace(/\/$/, "");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Avoid picking a parent-folder lockfile (e.g. C:\Users\...\package-lock.json) as workspace root on Windows.
  outputFileTracingRoot: path.join(__dirname),
  async rewrites() {
    return [{ source: "/documind-api/:path*", destination: `${apiTarget}/:path*` }];
  }
};

export default nextConfig;
