// Static export only: no API routes, no server actions, no middleware. FastAPI serves web/out
// at /admin-ui, so the pages call the API on the same origin.
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  basePath: "/admin-ui",
  trailingSlash: true,
  images: { unoptimized: true },
  eslint: { ignoreDuringBuilds: true }, // type errors still fail the build (tsc)
};

export default nextConfig;
