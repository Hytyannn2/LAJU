import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  // Designs (GA) and SUMO runs take minutes; the /engine proxy's default timeout is 30 s.
  experimental: { proxyTimeout: 600_000, agentUpgrade: "security" },
  async rewrites() {
    return [{ source: "/engine/:path*", destination: "http://127.0.0.1:8000/:path*" }];
  },
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
