# Releasing ragsage to PyPI

Releases are published by [`.github/workflows/release.yml`](../.github/workflows/release.yml)
when a `v*` tag is pushed. No API token exists anywhere: PyPI authenticates the workflow
itself through OIDC ([Trusted Publishing](https://docs.pypi.org/trusted-publishers/)).

Two facts shape everything below. **A version number on PyPI can never be reused**, even
after a deletion — `0.1.0` is spent the moment it uploads, mistakes included. And **the
sdist is public source**: publishing puts the code in front of anyone, whether or not the
GitHub repo is public.

## One-time setup

1. **Create the project's Trusted Publisher on PyPI.** `ragsage` is unclaimed, so use the
   [pending publisher](https://pypi.org/manage/account/publishing/) form — it reserves the
   name and lets the first upload create the project. Fill it in exactly:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `ragsage` |
   | Owner | `nirajk77777` |
   | Repository name | `ragsage` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   These must match the workflow character for character. A mismatch fails at the upload
   step with `invalid-publisher`, after the build has already passed.

2. **Create the `pypi` GitHub environment** (repo Settings → Environments → New). It needs
   no secrets — it exists so the publisher above can be scoped to it, and so you have a
   place to add required reviewers if you ever want a release to need a human click.

## Each release

1. Bump `version` in `pyproject.toml`. The workflow refuses to publish a tag that
   disagrees with it, so this is the single source of truth.
2. Commit, push to `main`, and let CI go green.
3. Tag and push:

   ```console
   $ git tag v0.1.0
   $ git push origin v0.1.0
   ```

4. Watch the run (`gh run watch`). The `build` job re-runs the full gate set — ruff, mypy,
   pytest, and the tag/version check — before the `publish` job, which is the only job
   holding the OIDC credential and runs no project code.

## Checking the result

```console
$ uv run --with ragsage --no-project -- ragsage --help
```

That resolves from PyPI in a throwaway environment, so it proves what a stranger gets
rather than what your working tree has.

## Verifying a build without publishing

Everything except the upload can be checked locally:

```console
$ uv build
$ uvx twine check --strict dist/*
$ tar -tzf dist/ragsage-*.tar.gz
```

The last command is worth actually reading. The sdist is assembled from the working tree,
not from git, so anything untracked-but-unignored lands in it — a stray agent worktree
under `.claude/` once added 76 files and doubled the archive. `[tool.hatch.build.targets.sdist]`
in `pyproject.toml` excludes the known offenders; the listing is how you catch the next one.
