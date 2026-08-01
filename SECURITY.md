# Security policy

## Reporting a vulnerability

**Please don't open a public issue for a security problem.**

Email **nirajk77777@gmail.com** with `[ragsage security]` in the subject. Include
enough to reproduce: the version, what you did, what happened, and what you expected.
A proof-of-concept helps but isn't required — a clear description of the flaw is
plenty to start.

What to expect, stated honestly rather than aspirationally: this is a single-maintainer
alpha project, not a funded security team. You should get an acknowledgement within a
week. If a report is valid I'll agree a disclosure timeline with you, credit you in the
advisory and the release notes unless you'd rather stay anonymous, and publish a fixed
version. If I judge a report out of scope I'll say so and explain why, rather than
leaving it unanswered.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ latest release only |
| < 0.1 | ❌ |

The package is alpha and pre-1.0. Fixes land in a new release on the current line;
there are no backports to older versions.

## What's in scope

The parts of ragsage where a flaw would cross a trust boundary:

- **Namespace isolation.** `Scope.namespace` is the only isolation boundary the engine
  has, and Postgres Row-Level Security enforces it. Anything that reads or writes chunks
  across namespaces — a store that forgets its predicate, a session that doesn't drop
  into the app role, an RLS policy that can be bypassed — is the highest-severity class
  of bug in this project.
- **SQL injection.** The app role, the isolation variable and the text-search
  configuration reach interpolated DDL because Postgres offers no bind form for them.
  They're guarded by `safe_identifier` / `safe_setting_name`. A way past those guards is
  in scope.
- **Credential handling.** API keys live in `ProviderConfig` and DSNs in
  `PostgresConfig`. A path that leaks either into logs, traces, exception messages or
  the docs is in scope.
- **Supply chain.** Releases publish through PyPI
  [Trusted Publishing](https://docs.pypi.org/trusted-publishers/): no API token exists
  anywhere, and the job holding the OIDC credential runs no project code. A flaw in
  that release path is in scope — see
  [releasing](https://docs.ragsage.163.128.113.41.sslip.io/releasing).
- **Untrusted document handling.** ragsage parses PDF, DOCX, PPTX and HTML supplied by
  whoever uploads them. A malicious document that achieves code execution, an
  unbounded resource exhaustion, or an SSRF/file read through a parser path is in
  scope.

## What's out of scope

Not because these don't matter, but because they're known, documented properties
rather than defects — reporting one won't get a fix, though a proposal to improve the
documentation is welcome:

- **Prompt injection through ingested content.** A retrieved chunk is untrusted text
  placed in a model's context, and text in your corpus can influence the answer
  generated from it. ragsage grounds answers in retrieved chunks and binds citations to
  them; it does not, and cannot, sanitise documents into being safe to read. Treat the
  corpus as a trust boundary you control.
- **Answer quality, hallucination and retrieval misses.** Ordinary bugs — file them as
  issues. See
  [failure modes](https://docs.ragsage.163.128.113.41.sslip.io/failure-modes) for the
  parser's known weak spots.
- **Vulnerabilities in dependencies**, unless ragsage's own use of one is what makes it
  exploitable. Report those upstream; tell me too if ragsage needs a version bump.
- **Anything requiring an attacker who already holds the database owner credentials or
  the provider API keys.** The owner role bypasses RLS by design — that's why scoped
  work drops into a non-privileged role first.
- **Misconfiguration by a consumer**, such as passing one tenant's namespace to another
  tenant's request. `Scope` is opaque to the engine on purpose; mapping a user to a
  namespace is the consumer's responsibility.

## Disclosure

I'll publish a GitHub Security Advisory when a fix ships, with a CVE if the severity
warrants one. Please give me a reasonable window to release before disclosing publicly
— and if I go quiet past the timeline we agreed, go ahead and publish. A dead
maintainer's inbox shouldn't be a reason a real flaw stays secret.
