/**
 * The documentation gate: assertions over the built static site.
 *
 * This runs inside the Dockerfile, on the very files nginx will serve, so the
 * thing CI checks and the thing that deploys are one artifact rather than two
 * that ought to agree. A failure here fails the build, and a failed build is not
 * deployed — the site cannot silently degrade into dead links.
 *
 * Every assertion is paired with a **canary** that establishes the corpus it ran
 * over was not empty. That is not ceremony. "Every internal link resolves" is
 * trivially true over zero links, and a vacuous green is exactly what hid the
 * generator defect this migration exists to fix: the builder emitted 367 anchored
 * links of which 3 resolved, and nothing looked wrong. A gate that cannot fail is
 * worse than no gate, because it is believed.
 *
 * Internal links only. External link rot needs the network and must never be able
 * to fail a deploy — the site does not become wrong when someone else's server
 * goes down.
 *
 *     node scripts/check-built-site.mjs out
 */

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative, posix, dirname } from 'node:path';
import process from 'node:process';

const outDir = process.argv[2] ?? 'out';
const contentDir = 'content';
const manifestPath = '.api-manifest.json';

/** Pages Next.js emits for its own purposes, which are not site content. */
const NON_CONTENT_PAGES = new Set(['/404', '/_not-found']);

/** Hrefs that are navigation-inert, and asset paths served straight from disk. */
const isIgnorableHref = (href) =>
  href === '' || href === '#' || href.startsWith('/_next/');

const failures = [];
const notes = [];

function check(condition, message) {
  if (!condition) failures.push(message);
}

/**
 * A canary: a claim about the corpus, not about the site.
 *
 * If one of these fails, the assertion it guards proved nothing — read it as
 * "the gate is broken", not "the site is broken".
 */
function canary(condition, message) {
  if (!condition) failures.push(`canary failed (the gate proved nothing): ${message}`);
}

// ---------------------------------------------------------------------------- #
// Read the built site
// ---------------------------------------------------------------------------- #

function walk(dir) {
  const found = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) found.push(...walk(full));
    else found.push(full);
  }
  return found;
}

/** `out/api/ports.html` is served at `/api/ports`; `out/index.html` at `/`. */
function routeOf(file) {
  const rel = relative(outDir, file).split('\\').join('/');
  const withoutSuffix = rel.slice(0, -'.html'.length);
  return withoutSuffix === 'index' ? '/' : `/${withoutSuffix}`;
}

const htmlFiles = walk(outDir).filter((file) => file.endsWith('.html'));
const pages = new Map();

for (const file of htmlFiles) {
  const route = routeOf(file);
  if (NON_CONTENT_PAGES.has(route)) continue;
  const html = readFileSync(file, 'utf8');
  pages.set(route, {
    html,
    // Only `<a href>`: `data-href` is used by the UI library for stylesheet
    // deduplication and points at nothing navigable.
    links: [...html.matchAll(/<a\b[^>]*?\shref="([^"]*)"/g)].map((match) => match[1]),
    ids: new Set([...html.matchAll(/\sid="([^"]*)"/g)].map((match) => match[1])),
  });
}

canary(pages.size > 0, `no pages were built into ${outDir}/`);

// ---------------------------------------------------------------------------- #
// 1. The expected pages exist at the expected routes
// ---------------------------------------------------------------------------- #

const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));

/** Content files, by the route they should have produced. */
function contentRoutes() {
  const routes = [];
  for (const file of walk(contentDir)) {
    if (!/\.mdx?$/.test(file)) continue;
    const rel = relative(contentDir, file).split('\\').join('/');
    const stem = rel.replace(/\.mdx?$/, '');
    routes.push(stem === 'index' ? '/' : `/${stem.replace(/\/index$/, '')}`);
  }
  return routes;
}

const expected = contentRoutes();
canary(expected.length > 0, `no content files found under ${contentDir}/`);
canary(
  manifest.routes.length > 0,
  'the generator manifest lists no API routes — was the generator run?',
);

for (const route of expected) {
  check(pages.has(route), `content exists for ${route} but no page was built for it`);
}
for (const route of manifest.routes) {
  check(pages.has(route), `the generator produced ${route} but no page was built for it`);
}
for (const route of pages.keys()) {
  check(
    expected.includes(route),
    `${route} was built but no content file corresponds to it`,
  );
}

// ---------------------------------------------------------------------------- #
// 2. Every internal link resolves, including its fragment
// ---------------------------------------------------------------------------- #

let checkedLinks = 0;
let checkedFragments = 0;

for (const [route, page] of pages) {
  for (const href of page.links) {
    if (isIgnorableHref(href) || /^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith('//')) {
      continue;
    }

    const [path, fragment] = href.split('#');
    // Relative hrefs resolve against the route, exactly as a browser resolves
    // them — which is why routes must not gain a trailing slash. `models` from
    // `/api/ports` is `/api/models`; from `/api/ports/` it would be
    // `/api/ports/models`, and every cross-page link in the reference would 404.
    const target =
      path === ''
        ? route
        : path.startsWith('/')
          ? path
          : posix.normalize(posix.join(dirname(route), path));

    checkedLinks += 1;

    const targetPage = pages.get(target === '/.' ? '/' : target);
    if (!targetPage) {
      failures.push(`${route}: link to ${href} resolves to ${target}, which is not a page`);
      continue;
    }

    if (fragment) {
      checkedFragments += 1;
      check(
        targetPage.ids.has(decodeURIComponent(fragment)),
        `${route}: link to ${href} resolves to ${target}, which has no #${fragment}`,
      );
    }
  }
}

canary(checkedLinks > 0, 'no internal links were found to check');
canary(
  checkedFragments > 100,
  `only ${checkedFragments} anchored links were checked; the API reference alone has hundreds`,
);
notes.push(`${checkedLinks} internal links (${checkedFragments} anchored) resolve`);

// ---------------------------------------------------------------------------- #
// 3. No signature carries the eaten keyword-only marker
// ---------------------------------------------------------------------------- #

const apiPages = [...pages].filter(([route]) => route === '/api' || route.startsWith('/api/'));
canary(apiPages.length > 0, 'no API reference pages were built');

let restored = 0;
for (const [route, page] of apiPages) {
  const artefacts = page.html.match(/, , |\(, /g);
  check(
    artefacts === null,
    `${route}: ${artefacts?.length} signature(s) still show the eaten keyword-only marker`,
  );
  restored += (page.html.match(/, \*, |\(\*, /g) ?? []).length;
}

canary(
  restored >= manifest.markersRestored,
  `the generator restored ${manifest.markersRestored} keyword-only markers but only ${restored} ` +
    'reached the built site, so "no artefacts" may just mean "no signatures"',
);
notes.push(`${restored} keyword-only markers survive into the built pages`);

// ---------------------------------------------------------------------------- #
// 4. Every documented object still has its source link
// ---------------------------------------------------------------------------- #

const sourceLinkPattern = /^https:\/\/github\.com\/[^/]+\/[^/]+\/blob\/[^/]+\/src\//;
const servedSourceLinks = apiPages
  .flatMap(([, page]) => page.links)
  .filter((href) => sourceLinkPattern.test(href)).length;

canary(manifest.sourceLinks > 0, 'the generator injected no source links');
check(
  servedSourceLinks === manifest.sourceLinks,
  `the generator injected ${manifest.sourceLinks} source links but the built site serves ` +
    `${servedSourceLinks}`,
);
notes.push(`${servedSourceLinks} source links served`);

// ---------------------------------------------------------------------------- #
// 5. Navigation and content agree
// ---------------------------------------------------------------------------- #

const navigated = new Set();
const metaFiles = walk(contentDir).filter((file) => file.endsWith('meta.json'));
canary(metaFiles.length > 0, 'no meta.json navigation files were found');

for (const file of metaFiles) {
  const dir = dirname(relative(contentDir, file)).split('\\').join('/');
  const prefix = dir === '.' ? '' : `/${dir}`;
  const meta = JSON.parse(readFileSync(file, 'utf8'));

  for (const entry of meta.pages ?? []) {
    // Separators (`---Guides---`) and external links are labels, not pages.
    if (entry.startsWith('---') || entry.includes(':')) continue;

    const asPage = entry === 'index' ? prefix || '/' : `${prefix}/${entry}`;
    const isFolder = metaFiles.some(
      (other) => dirname(relative(contentDir, other)) === (dir === '.' ? entry : `${dir}/${entry}`),
    );

    if (isFolder) continue; // its own meta.json is checked on its own terms
    check(pages.has(asPage), `navigation lists "${entry}" in ${file}, but ${asPage} is not a page`);
    navigated.add(asPage);
  }
}

for (const route of pages.keys()) {
  check(navigated.has(route), `${route} was built but no meta.json navigates to it`);
}

canary(navigated.size > 0, 'navigation lists no pages at all');

// ---------------------------------------------------------------------------- #

for (const note of notes) console.log(`  ok  ${note}`);

if (failures.length > 0) {
  console.error(`\n${failures.length} problem(s) with the built site:\n`);
  for (const failure of failures) console.error(`  - ${failure}`);
  console.error('');
  process.exit(1);
}

console.log(`\nthe built site is sound: ${pages.size} pages checked.`);
