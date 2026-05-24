# ADR 003: Tag normalization policy (v0.1)

## Status

Accepted (v0.1)

## Context

Keyword fields must be deterministic and deduplicated before persistence. Singularization and synonym maps need a curated source of truth and are deferred.

## Decision

Normalization runs **once** in `clip-tag-core::normalize` before write planning:

1. Trim leading/trailing whitespace.
2. Unicode casefold to lowercase.
3. Collapse internal separators (`-`, `_`, `/`, `,`, whitespace) to a single space.
4. Strip leading/trailing ASCII punctuation (except `#` and `+` for common photo tokens).
5. Dedupe with stable lexicographic ordering (`BTreeSet`).

**Out of scope for v0.1:**

- English singularization / plural stripping.
- Synonym or alias mapping (planned with vocabulary supply chain in a later phase).

Supervisor decision (2026-05-24): preserve scored vocabulary strings; no singularization in v0.1.

## Consequences

- Writers must not apply per-format normalization.
- Golden tests for normalization are unit-tested in `clip-tag-core`.
