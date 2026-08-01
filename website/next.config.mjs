import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/**
 * Static export. The production image is nginx over a directory of files — no
 * Node runtime, no Python, nothing to keep alive. That constraint is what rules
 * out `redirects`, which static export cannot honour: a URL can only be retired
 * at the proxy, or by leaving static export.
 *
 * `trailingSlash` is deliberately left off, so a page exports as `quickstart.html`
 * rather than `quickstart/index.html`. nginx is configured to try `$uri.html`;
 * the two halves have to agree, and this is the half that decides.
 *
 * @type {import('next').NextConfig}
 */
const config = {
  output: 'export',
  reactStrictMode: true,
};

export default withMDX(config);
