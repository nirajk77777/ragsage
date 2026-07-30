# Releasing ragsage to PyPI

Releases are published by [`.github/workflows/release.yml`](https://github.com/nirajk77777/ragsage/blob/main/.github/workflows/release.yml)
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
2. *Optional* — write `docs/release-notes/v<version>.md`. See "Release notes" below.
3. Commit, push to `main`, and let CI go green.
4. Tag and push:

   ```console
   $ git tag v0.1.0
   $ git push origin v0.1.0
   ```

5. Watch the run (`gh run watch`). Three jobs, in order:

   | Job | Does | Holds |
   | --- | --- | --- |
   | `build` | ruff, mypy, pytest, tag/version check, `uv build` | nothing |
   | `publish` | uploads to PyPI | the OIDC credential, and runs no project code |
   | `github-release` | creates the GitHub release, attaches the artifacts | `contents: write` |

   `github-release` runs only after PyPI accepts the upload, so a GitHub release never
   advertises a version nobody can install. The attached files are the `build` job's own
   output rather than a rebuild, so they cannot drift from what PyPI serves.

## Release notes

`github-release` picks notes in this order:

1. **`docs/release-notes/v<version>.md`**, if it exists — verbatim.
2. Otherwise **`--generate-notes`**, which is GitHub's commit-and-PR list since the last
   tag.

The generated list is the right summary for a patch release and the wrong one for a
release that needs explaining, which is why the file is the override rather than the only
option.
[`v0.1.0.md`](https://github.com/nirajk77777/ragsage/blob/main/docs/release-notes/v0.1.0.md)
is the worked example.

A tag carrying a PEP 440 pre-release segment — `v1.0.0rc1`, `v0.2.0a1`, `v0.3.0.dev1` — is
marked pre-release and does *not* move GitHub's "Latest", matching how PyPI treats it.

Re-running a release workflow on an existing tag **refreshes the attached assets and
leaves the notes alone**, so prose edited on the release page survives.

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
