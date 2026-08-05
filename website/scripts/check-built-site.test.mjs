/**
 * The gate's own gate: proof that `check-built-site.mjs` fails when it should.
 *
 *     npm run build && npm test
 *
 * A build gate is believed. That is its entire value and its entire danger: once
 * it is green nobody looks at the site again, so a gate that has quietly stopped
 * being able to fail is worse than no gate at all — it converts an unchecked site
 * into one everybody thinks is checked. This migration already met that failure
 * mode once, in the generator: it emitted 367 anchored links of which 3 resolved,
 * and every source-level check passed over it.
 *
 * So each check earns its place here by being *shown* to fail. Every test below
 * takes the site as it was actually built, breaks exactly one thing, and asserts
 * that the gate reports that one thing — by name. Asserting a non-zero exit alone
 * would be the same mistake one level up: a gate broken in some unrelated way
 * exits non-zero for every mutation, and this file would go green over it.
 *
 * **Mutations are applied to the real export**, not to a hand-written fixture. A
 * fixture would be a second model of what Next.js emits, maintained here, drifting
 * from the first — and the day the layout stops wrapping pages in `<article>` or
 * the search index changes shape, the fixture keeps these tests green while the
 * gate silently ranges over nothing. The corpus under test has to be the artifact,
 * for the same reason the gate checks the built site rather than the sources.
 *
 * That makes `out/` a precondition rather than an input, which is why this file
 * refuses to run without one: skipping would be indistinguishable from passing.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import {
  cpSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import process from 'node:process';

const WEBSITE = dirname(dirname(fileURLToPath(import.meta.url)));
const GATE = join(WEBSITE, 'scripts', 'check-built-site.mjs');

/** The three things the gate reads: the export, the sources, the generator's report. */
const INPUTS = [
  ['out', join(WEBSITE, 'out')],
  ['content', join(WEBSITE, 'content')],
  ['.api-manifest.json', join(WEBSITE, '.api-manifest.json')],
];

for (const [name, path] of INPUTS) {
  if (!existsSync(path)) {
    throw new Error(
      `${name} is missing, so there is no built site to break. These tests measure the gate ` +
        'against the artifact it gates rather than against a fixture, so they cannot run ' +
        'without one: `npm run build` first (and `python tools/generate_api_docs.py` before ' +
        'that, if content/api/ is what is missing).',
    );
  }
}

// ---------------------------------------------------------------------------- #
// Running the gate over a copy of the site, with one thing broken
// ---------------------------------------------------------------------------- #

/**
 * Run the gate over a private copy of the built site, mutated on the way in.
 *
 * A copy per test, because the gate reads the same tree the site is served from
 * and a mutation left behind would be a broken deploy. The copy is cheap — the
 * export is a hundred-odd files — and it is what lets every test below describe
 * its own breakage in full rather than undoing the last one.
 */
function gateOver(mutate = () => {}) {
  const dir = mkdtempSync(join(tmpdir(), 'ragsage-gate-'));
  try {
    for (const [name, path] of INPUTS) cpSync(path, join(dir, name), { recursive: true });
    mutate(dir);

    const run = spawnSync(process.execPath, [GATE, 'out'], { cwd: dir, encoding: 'utf8' });
    return { code: run.status, output: `${run.stdout ?? ''}${run.stderr ?? ''}` };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/**
 * The gate failed, and the mutation is the *whole* reason it failed.
 *
 * The problem count is what makes this a statement about the mutation rather than
 * about the gate's mood. One mutation, one problem: a gate that had started
 * reporting a second, unrelated failure on every run would satisfy "exits
 * non-zero" and "mentions the link" forever, and this file would never notice.
 */
function assertSoleProblem({ code, output }, ...fragments) {
  assert.equal(code, 1, `the gate passed over a site that is broken:\n${output}`);
  assert.match(
    output,
    /\n1 problem\(s\) with the built site/,
    `one thing was broken, but the gate reported more or fewer:\n${output}`,
  );
  for (const fragment of fragments) {
    assert.ok(
      output.includes(fragment),
      `the gate failed, but never said "${fragment}":\n${output}`,
    );
  }
}

/** The gate failed, and said these things among others. */
function assertReports({ code, output }, ...fragments) {
  assert.equal(code, 1, `the gate passed over a site that is broken:\n${output}`);
  for (const fragment of fragments) {
    assert.ok(
      output.includes(fragment),
      `the gate failed, but never said "${fragment}":\n${output}`,
    );
  }
}

// ---------------------------------------------------------------------------- #
// Breaking one thing
// ---------------------------------------------------------------------------- #

/**
 * Put a link on a page's own content, where a page of prose carries one.
 *
 * Injected rather than rewritten, so the test knows exactly what it broke: the
 * export's existing hrefs are the generator's and the layout's, and editing one
 * would make the expected message depend on whichever link happened to be first.
 *
 * Into the `<article>`, because that is the page rather than the layout wrapped
 * around it — and it asserts it got there. A mutation that lands nowhere leaves a
 * test asserting that an unbroken site fails, which is a confusing way to
 * discover that the layout changed.
 */
function linkOn(page, href) {
  return (dir) => {
    const file = join(dir, 'out', `${page}.html`);
    const html = readFileSync(file, 'utf8');
    assert.ok(
      html.includes('</article>'),
      `${page} carries no <article>, so this mutation would have gone nowhere`,
    );
    writeFileSync(file, html.replace('</article>', `<a href="${href}">a link</a></article>`));
  };
}

/** Rewrite one of the site's JSON inputs — navigation, or the generator's report. */
function edit(relative, change) {
  return (dir) => {
    const file = join(dir, relative);
    const value = JSON.parse(readFileSync(file, 'utf8'));
    writeFileSync(file, JSON.stringify(change(value) ?? value, null, 2));
  };
}

/** Every file under a directory, recursively. */
function filesUnder(dir) {
  return readdirSync(dir).flatMap((entry) => {
    const full = join(dir, entry);
    return statSync(full).isDirectory() ? filesUnder(full) : [full];
  });
}

/** Apply several mutations to one copy of the site. */
const all =
  (...mutations) =>
  (dir) => {
    for (const mutate of mutations) mutate(dir);
  };

// ---------------------------------------------------------------------------- #

describe('the gate over the site as built', () => {
  it('passes', () => {
    const { code, output } = gateOver();

    // The canary for every test below. Each of them asserts that a mutated site
    // fails; that means nothing unless the unmutated one passes, because a site
    // already failing would fail after any mutation whatsoever — including a
    // mutation the gate cannot actually see.
    assert.equal(
      code,
      0,
      'the built site does not pass its own gate, so no test in this file proves anything ' +
        `about the mutation it makes. Fix the site first — "npm run check":\n${output}`,
    );
  });
});

describe('a link that leads nowhere', () => {
  it('fails the build, naming the link and the page carrying it', () => {
    const result = gateOver(linkOn('quickstart', '/no-such-page'));

    // Both halves, because either alone leaves the reader bisecting: the site has
    // 1700 links and the message has to say which one, on which of eleven pages.
    assertSoleProblem(result, '/quickstart:', '/no-such-page');
  });

  it('fails when the page resolves but the fragment does not', () => {
    const result = gateOver(linkOn('quickstart', '/api/ports#no-such-anchor'));

    // The defect this migration exists to fix, in its exact shape: a link whose
    // page is real and whose anchor is not. It renders, it navigates, it lands on
    // the wrong part of a page that happens to exist — and a checker that stops
    // at the path calls it healthy. 364 of this site's 367 anchored links were
    // this, and nothing reported it.
    assertSoleProblem(result, '/quickstart:', '/api/ports#no-such-anchor', 'no-such-anchor');
  });

  it('says nothing about a link that leaves the site', () => {
    const { code, output } = gateOver(linkOn('quickstart', 'https://example.invalid/gone'));

    // External link rot is real and is not this gate's business: it needs the
    // network, it is nobody's regression, and a deploy that fails because someone
    // else's server is down is a deploy that gets its gate switched off.
    assert.equal(code, 0, `the gate followed a link off the site:\n${output}`);
    assert.ok(
      !output.includes('example.invalid'),
      `the gate had something to say about an external link:\n${output}`,
    );
  });
});

describe('navigation and content disagreeing', () => {
  it('fails when a page is orphaned from navigation', () => {
    const result = gateOver(
      edit('content/meta.json', (meta) => ({
        ...meta,
        pages: meta.pages.filter((page) => page !== 'quickstart'),
      })),
    );

    // A page nothing navigates to is not published, it is merely reachable — by
    // search, or by a link someone already has. Nothing else here would report
    // it: the page builds, its links resolve, and every link *to* it still works.
    assertSoleProblem(result, '/quickstart', 'no meta.json navigates to it');
  });

  it('fails when navigation points at a file that is not there', () => {
    const result = gateOver(
      edit('content/meta.json', (meta) => ({ ...meta, pages: [...meta.pages, 'no-such-page'] })),
    );

    // The other direction, and the one a reader meets first: a sidebar entry that
    // 404s. Fumadocs drops an unresolvable entry silently rather than failing the
    // build, so without this the entry simply stops appearing.
    assertSoleProblem(result, 'no-such-page', 'content/meta.json');
  });
});

describe('a build with less in it than it should have', () => {
  it('reports an empty build as a failure rather than a vacuous success', () => {
    const result = gateOver(
      all(
        (dir) => {
          for (const file of filesUnder(join(dir, 'out'))) rmSync(file);
          for (const file of filesUnder(join(dir, 'content'))) rmSync(file);
        },
        edit('.api-manifest.json', (manifest) => ({ ...manifest, routes: [] })),
      ),
    );

    // Everything this gate asserts is trivially true of a site with no pages in
    // it: every link resolves, every signature is intact, every navigation entry
    // agrees with every page. That is the failure this whole design is arranged
    // around — a build that half-failed, deployed green, and took the site down
    // while its gate said nothing.
    assertReports(
      result,
      'no pages were built into out/',
      'no content files found under content/',
      'the generator manifest lists no API routes',
      'no internal links were found to check',
      'no API reference pages were built',
      'no meta.json navigation files were found',
      'navigation lists no pages at all',
      'no front page was built',
      'no hand-written pages were built',
    );

    // And named as what they are. A canary firing means the gate proved nothing,
    // which is a different thing to tell a contributor than "the site is broken"
    // — they would go looking at the site.
    const canaries = result.output.match(/canary failed \(the gate proved nothing\)/g) ?? [];
    assert.ok(canaries.length >= 9, `only ${canaries.length} canaries fired:\n${result.output}`);
  });

  it('fails when the API reference is missing, though the prose still builds', () => {
    const result = gateOver((dir) => {
      rmSync(join(dir, 'out', 'api.html'));
      for (const file of filesUnder(join(dir, 'out', 'api'))) {
        if (file.endsWith('.html')) rmSync(file);
      }
    });

    // The partial build that a whole-corpus canary would sail straight past. Five
    // prose pages still carry links, still carry articles, still navigate — so
    // "some pages were built" and "some links were checked" both hold, and the
    // six pages that hold the entire API surface are gone. The canaries that
    // matter are the ones scoped to the corpus whose claim they guard.
    assertReports(
      result,
      'no page was built for it',
      'no API reference pages were built',
      'anchored links were checked',
    );
  });

  it('fails when a single page did not build', () => {
    const result = gateOver((dir) => rmSync(join(dir, 'out', 'quickstart.html')));

    // The narrowest partial build there is: one page, of the eleven, absent from
    // the export. Named against the content file that should have produced it, so
    // the message points at a cause rather than at a symptom.
    assertReports(result, 'content exists for /quickstart but no page was built for it');
  });
});
