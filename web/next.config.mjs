import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Avoid picking a parent-folder lockfile (e.g. C:\Users\...\package-lock.json) as workspace root on Windows.
  outputFileTracingRoot: path.join(__dirname)
};

export default nextConfig;
