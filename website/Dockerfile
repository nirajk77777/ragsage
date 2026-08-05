# The documentation site: generate, build, gate, serve.
#
# Three stages, and only the last one ships. Python generates the API reference
# from the docstrings in `src/`; Node builds a static export and refuses to
# proceed if the result has a dead link; nginx serves the files. Neither Python
# nor Node exists in the image that runs — the deployed surface is static files
# and a web server, which is the whole reason the site is built this way.
#
# **Build context is the repository root**, not `website/`. It has to be: the
# generator imports `ragsage` and reads `src/`, and a context rooted at `website/`
# could not see either. On Coolify that means leaving Base Directory as `/` and
# pointing Dockerfile Location at this file — the two settings are not
# interchangeable, because Coolify derives the build context from Base Directory.
#
#     docker build -f website/Dockerfile -t ragsage-docs .

# ---------------------------------------------------------------------------- #
# Stage 1 — Sphinx reads the docstrings, and Markdown comes out
# ---------------------------------------------------------------------------- #
FROM python:3.12-slim AS api-reference

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /repo

# The lockfile and the project metadata first, so a docstring edit does not
# invalidate the dependency layer.
#
# `--no-install-project` is what makes that true. The API reference is generated
# *from* `src/`, so `src/` is this stage's most-edited input by far — and with the
# project installed in the same step, every docstring edit re-resolved Sphinx and
# its dependency tree. Resolution depends on the lockfile alone, so it belongs in
# a layer only the lockfile invalidates. The second sync then installs the project
# itself against dependencies already present, which is a build, not a resolve.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --group docs --no-dev --no-install-project

COPY src/ src/
RUN uv sync --locked --group docs --no-dev

COPY api-src/ api-src/
COPY tools/ tools/

# Writes `website/content/api/` and `website/.api-manifest.json`. It fails loudly
# on a Sphinx warning that is not one of the builder defects it corrects, so a
# broken cross-reference stops the build here rather than becoming a dead link
# three stages later.
RUN uv run --group docs python tools/generate_api_docs.py

# ---------------------------------------------------------------------------- #
# Stage 2 — Next.js builds a static export, and the gate inspects it
# ---------------------------------------------------------------------------- #
FROM node:22-slim AS site

WORKDIR /website

COPY website/package.json website/package-lock.json ./
RUN npm ci

COPY website/ ./
COPY --from=api-reference /repo/website/content/api/ ./content/api/
COPY --from=api-reference /repo/website/.api-manifest.json ./.api-manifest.json

RUN npm run build

# The gate. It walks the files nginx is about to serve and asserts every internal
# link resolves down to its fragment, that no signature lost its keyword-only
# marker, that every source link the generator injected is still there, and that
# navigation and content are in bijection — each paired with a canary, so an empty
# or partial build reports failure rather than a vacuous success.
RUN npm run check

# And the gate's own gate: the export is copied, one thing is broken in the copy,
# and the gate is required to report that one thing by name. A build gate is
# believed, which is exactly what makes a gate that has quietly lost the ability
# to fail worse than none at all.
#
# It runs here, after the export exists, because it measures the gate against the
# real artifact rather than a fixture — a fixture would be a second model of what
# Next.js emits, and it would drift. That puts it in this stage rather than in the
# Python test suite, which has no `out/` to break.
RUN npm test

# ---------------------------------------------------------------------------- #
# Stage 3 — nginx, and nothing else
# ---------------------------------------------------------------------------- #
FROM nginx:1.27-alpine AS runtime

COPY website/nginx.conf /etc/nginx/conf.d/default.conf

# Its own directory, rather than the document root nginx already populates, so
# that what is served is the export and nothing else. `nginx.conf` names the same
# path; the two are one decision, and `tests/test_documentation_image.py` keeps
# them from drifting apart, since nothing brings them together until build time.
COPY --from=site /website/out/ /srv/site/

EXPOSE 80
