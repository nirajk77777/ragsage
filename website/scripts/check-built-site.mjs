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

/**
 * Every link in a fragment of markup.
 *
 * Only `<a href>`: `data-href` is used by the UI library for stylesheet
 * deduplication, and React hoists stylesheets by putting an `href` on a `<style>`
 * tag. Neither points at anything navigable.
 */
function hrefsIn(html) {
  return [...html.matchAll(/<a\b[^>]*?\shref="([^"]*)"/g)].map((match) => match[1]);
}

for (const file of htmlFiles) {
  const route = routeOf(file);
  if (NON_CONTENT_PAGES.has(route)) continue;
  const html = readFileSync(file, 'utf8');
  // The page's own content. Everything the layout wraps around it — the sidebar,
  // the table of contents — is the site's, and belongs to no page in particular.
  // Empty if the layout ever stops using `<article>`, which is what the canaries
  // over this corpus are there to notice.
  const article = /<article\b[\s\S]*?<\/article>/.exec(html)?.[0] ?? '';
  pages.set(route, {
    html,
    article,
    links: hrefsIn(html),
    ids: new Set([...html.matchAll(/\sid="([^"]*)"/g)].map((match) => match[1])),
    headingIds: [...article.matchAll(/<h[2-6]\b[^>]*\sid="([^"]*)"/g)].map((match) => match[1]),
  });
}

/** Whether an already-resolved page carries the anchor a link or result names. */
const hasAnchor = (page, fragment) => page.ids.has(decodeURIComponent(fragment));

/** The generated reference: `/api` is the folder's own page, not a page above it. */
const isApiRoute = (route) => route === '/api' || route.startsWith('/api/');

/** Whether an href points somewhere on this site at all. */
const isInternalHref = (href) =>
  !isIgnorableHref(href) && !/^[a-z][a-z0-9+.-]*:/i.test(href) && !href.startsWith('//');

/**
 * The route an href on `from` leads to, exactly as a browser would resolve it —
 * which is why routes must not gain a trailing slash. `models` from `/api/ports`
 * is `/api/models`; from `/api/ports/` it would be `/api/ports/models`, and every
 * cross-page link in the reference would 404.
 */
function resolveHref(from, href) {
  const [path, fragment] = href.split('#');
  const target =
    path === ''
      ? from
      : path.startsWith('/')
        ? path
        : posix.normalize(posix.join(dirname(from), path));
  return { target: target === '/.' ? '/' : target, fragment };
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
    if (!isInternalHref(href)) continue;

    const { target, fragment } = resolveHref(route, href);
    checkedLinks += 1;

    const targetPage = pages.get(target);
    if (!targetPage) {
      failures.push(`${route}: link to ${href} resolves to ${target}, which is not a page`);
      continue;
    }

    if (fragment) {
      checkedFragments += 1;
      check(
        hasAnchor(targetPage, fragment),
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

const apiPages = [...pages].filter(([route]) => isApiRoute(route));
canary(apiPages.length > 0, 'no API reference pages were built');

/**
 * Signatures, and nothing else on the page.
 *
 * The generator emits every signature as a Markdown heading and confines its
 * repair to heading lines, on the grounds that a docstring is free to contain
 * `, ,` in a prose aside or a code sample — worth leaving alone rather than
 * corrupting prose to fix signatures. Scanning whole-page HTML here would put
 * this gate in direct conflict with that: a docstring exercising the exemption
 * would fail the deploy with nothing able to correct it.
 *
 * Scoping to `h2`–`h6` matches the generator's own `^#{2,6} ` filter, so the two
 * range over the same corpus and the counts below can be compared exactly. Page
 * titles are lifted into frontmatter and render as `h1`, which holds no
 * signature; the rendered payload repeats each heading roughly four times, in
 * the table of contents and the router data, which is what made the old
 * whole-page count read 140 against a manifest of 35.
 */
const signatureHeadings = ([, page]) =>
  [...page.html.matchAll(/<h([2-6])\b[^>]*>([\s\S]*?)<\/h\1>/g)].map((match) => match[0]);

const headingCount = apiPages.reduce((total, entry) => total + signatureHeadings(entry).length, 0);
canary(headingCount > 0, 'no headings were found on the API pages, so no signature was scanned');
canary(manifest.markersRestored > 0, 'the generator restored no keyword-only markers');

let restored = 0;
for (const entry of apiPages) {
  const [route] = entry;
  const headings = signatureHeadings(entry).join('\n');
  const artefacts = headings.match(/, , |\(, /g);
  check(
    artefacts === null,
    `${route}: ${artefacts?.length} signature(s) still show the eaten keyword-only marker`,
  );
  restored += (headings.match(/, \*, |\(\*, /g) ?? []).length;
}

// Exactly, not at least. Both corpora are now the same set of signatures, so a
// drift either way is a defect: too few means markers were lost between the
// generator and the served page, too many means something else is being counted.
check(
  restored === manifest.markersRestored,
  `the generator restored ${manifest.markersRestored} keyword-only markers but ${restored} ` +
    'appear in the built signatures',
);
notes.push(`${restored} keyword-only markers survive into the built signatures`);

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

// The front page's job is to get a first-time reader running, and the sidebar is
// not that: it lists everything at once and says nothing about where to start.
// Navigation being *complete* is what the checks above establish, which is
// exactly the state in which this one can quietly stop being true.
//
// Resolved rather than string-matched, so a relative `quickstart` counts — the
// question is where the reader lands, not how the href was spelled.
const frontPage = pages.get('/');
canary(frontPage !== undefined, 'no front page was built, so nothing was asked about it');
check(
  frontPage?.links
    .filter(isInternalHref)
    .some((href) => resolveHref('/', href).target === '/quickstart'),
  'the front page carries no link to /quickstart, so the first thing to do is a page away',
);

// ---------------------------------------------------------------------------- #
// 6. No page still shows the dialect it was converted from
// ---------------------------------------------------------------------------- #

/**
 * The prose was MyST inside a Sphinx tree, and the reference is generated from
 * docstrings that are still written in Sphinx's dialect. Both arrive here through
 * a translation, and both translations fail the same way: the syntax survives as
 * literal text. A page reading ``{class}`Scope``` or ``:::{note}`` is not a
 * broken link and not a missing page — it renders, it is navigable, it is
 * indexed, and every other check here passes over it.
 *
 * This is the one check that would have to change the day a page needs to *show*
 * this syntax rather than use it. That page would be documenting Sphinx, and it
 * would need somewhere to say so; nothing here is that page today.
 */
/**
 * Only the artefacts that can reach a built page. Two of the four shapes the
 * conversion had to remove cannot: a brace in `.mdx` is a JavaScript expression,
 * so a surviving `{class}` fails the MDX parse, and a fenced ```` ```{toctree} ````
 * fails Shiki, which refuses a language it does not have. Both were tried against
 * this site. What is left are the two that render quietly — a role on a generated
 * `.md` page, where braces stay literal, and a colon-fenced block anywhere.
 *
 * Each pattern is anchored on what the syntax renders *as*, not on the words in
 * it: a role survives as its brace or colon pair pressed against the code span it
 * was meant to link, `{class}<code>Scope</code>`. Requiring that neighbour is what
 * separates a leftover from a page that says `{data}` in a template or writes a
 * sentence about `:maxdepth:` — a page about this migration would do both.
 */
const DIALECT_ARTEFACTS = [
  ['a MyST cross-reference role', /\{(?:py:)?(?:class|func|meth|mod|attr|data|exc|obj|const|ref|doc|term)\}(?=<code|`)/],
  ['a Sphinx cross-reference role', /:(?:py:)?(?:class|func|meth|mod|attr|data|exc|obj|const|ref|doc|term):(?=<code|`)/],
  ['a MyST directive block', /:::\{[a-z]/],
];

const prosePages = [...pages].filter(([route]) => !isApiRoute(route));
const contentless = [...pages].filter(([, page]) => page.article === '').map(([route]) => route);

// Two canaries, because two things can empty this corpus. A layout that stops
// wrapping content in `<article>` leaves every page scanning an empty string,
// which passes silently for every pattern on every page — and it is a layout
// change, so nothing about the content would look wrong that day. Per page rather
// than in total: one page's worth of markup is enough to satisfy any threshold
// while the other ten scan nothing.
canary(
  prosePages.length > 0,
  'no hand-written pages were built, so nothing converted from MyST was scanned',
);
canary(
  contentless.length === 0,
  `no <article> content was found to scan on ${contentless.length} page(s): ${contentless.slice(0, 5)}`,
);

for (const [route, page] of pages) {
  for (const [label, pattern] of DIALECT_ARTEFACTS) {
    const found = pattern.exec(page.article);
    check(
      found === null,
      `${route}: rendered page still shows ${label} — "${page.article
        .slice(Math.max(0, (found?.index ?? 0) - 40), (found?.index ?? 0) + 60)
        .replace(/<[^>]*>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()}"`,
    );
  }
}

// The prose is where the roles were, and it is also where a lost cross-reference
// is invisible: the sentence still reads. Section 2 proves these links resolve —
// this proves there are still links to resolve. Both directions, because the
// reference contributes some 1700 links of its own and would carry section 2's
// canaries alone while every link between prose pages quietly went missing.
// From the article, not the whole page: the sidebar links every page to every
// other one and is on all of them, so counting it would answer this question with
// the navigation's links no matter what the prose itself said.
const proseLinks = prosePages.flatMap(([route, page]) =>
  hrefsIn(page.article)
    .filter(isInternalHref)
    .map((href) => ({ from: route, ...resolveHref(route, href) })),
);
const intoReference = proseLinks.filter((link) => isApiRoute(link.target) && link.fragment).length;
// To another page, not to a heading on this one. Every page carries a link per
// heading, so counting those would keep this canary green over prose that had
// stopped referring to itself entirely.
const betweenProse = proseLinks.filter(
  (link) => !isApiRoute(link.target) && link.target !== link.from,
).length;

canary(
  intoReference > 0,
  'no prose page links into the API reference, so section 2 proved nothing about prose',
);
canary(betweenProse > 0, 'no prose page links to another, so section 2 proved nothing about them');
notes.push(
  `${pages.size} pages carry no unconverted syntax (${prosePages.length} of them hand-written, ` +
    `linking ${betweenProse} times to each other and ${intoReference} into the reference)`,
);

// ---------------------------------------------------------------------------- #
// 7. Search reaches the generated reference
// ---------------------------------------------------------------------------- #

/**
 * The prebuilt Orama index, exported as a file because the production image has
 * no runtime to answer a query with.
 *
 * Nothing else here would notice search breaking. `staticGET` becoming `GET`, or
 * a content source dropping out of the loader, leaves every page still built,
 * still linked and still navigable — and the symbols on it findable only by
 * someone who already knows which page holds them.
 *
 * Every heading, not every page. A page contributes its own title to the index
 * whatever else goes wrong, so "each route appears" would pass over a reference
 * indexed down to nothing but six page names. Conservation is the same standard
 * sections 3 and 4 hold the generator to: what the site serves, the index knows.
 */
const searchIndexPath = join(outDir, 'api', 'search');

let searchDocs = null;
try {
  const exported = JSON.parse(readFileSync(searchIndexPath, 'utf8'));
  searchDocs = Object.values(exported.docs?.docs ?? {}).filter(Boolean);
} catch (error) {
  failures.push(`the search index at ${searchIndexPath} could not be read: ${error.message}`);
}

if (searchDocs !== null) {
  canary(searchDocs.length > 0, 'the search index holds no documents');

  /** Indexed headings, by the route they were indexed under. */
  const indexedHeadings = new Map();
  for (const entry of searchDocs) {
    if (entry.type !== 'heading') continue;
    const [route, fragment] = entry.url.split('#');
    if (!fragment) continue;
    if (!indexedHeadings.has(route)) indexedHeadings.set(route, new Set());
    indexedHeadings.get(route).add(decodeURIComponent(fragment));
  }

  let apiHeadings = 0;
  for (const [route, page] of pages) {
    const indexed = indexedHeadings.get(route) ?? new Set();
    if (isApiRoute(route)) apiHeadings += page.headingIds.length;

    for (const id of page.headingIds) {
      check(
        indexed.has(id),
        `${route} serves a heading at #${id} that search does not index, so the symbol it ` +
          'names can only be found by a reader who already knows the page',
      );
    }
  }

  // Scoped to the reference, because that is the corpus the claim is about: the
  // prose pages would carry this section past a run that indexed no API page at
  // all, and a symbol on a generated page is precisely what a reader searches for.
  canary(apiHeadings > 100, `only ${apiHeadings} headings were served by the API reference`);

  // A result that lands nowhere is worse than no result — the reader concludes
  // the symbol is undocumented. Slugs the site generates, a different set from
  // the `<a id>` targets section 2 ranges over.
  let checkedResultFragments = 0;
  for (const entry of searchDocs) {
    const [route, fragment] = entry.url.split('#');
    if (!fragment) continue;

    checkedResultFragments += 1;
    const targetPage = pages.get(route);
    if (!targetPage) {
      failures.push(`search result ${entry.url} points at ${route}, which is not a page`);
      continue;
    }

    check(
      hasAnchor(targetPage, fragment),
      `search result ${entry.url} points at ${route}, which has no #${fragment}`,
    );
  }

  notes.push(
    `${searchDocs.length} search entries index every served heading ` +
      `(${checkedResultFragments} anchored results resolve)`,
  );
}

// ---------------------------------------------------------------------------- #

for (const note of notes) console.log(`  ok  ${note}`);

if (failures.length > 0) {
  console.error(`\n${failures.length} problem(s) with the built site:\n`);
  for (const failure of failures) console.error(`  - ${failure}`);
  console.error('');
  process.exit(1);
}

console.log(`\nthe built site is sound: ${pages.size} pages checked.`);
