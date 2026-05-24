# ADR 002: Empty-only default write behavior

## Status

Accepted (v0.1)

## Context

Interop targets (Lightroom, Apple Photos, Capture One, Finder) may already populate keyword fields. Accidental merges can create duplicate or conflicting taxonomies.

## Decision

- Default write mode: **add tags only when all target keyword fields are empty** for the image.
- If any mapped field is non-empty and `--force` is not set, the policy engine returns `Skip` with an explicit reason; no file mutation occurs.
- Tagging (inference) may run without `--write-metadata`; **no metadata mutation** happens unless write mode is explicitly enabled.

## Consequences

- Dry-run and real-write paths share the same policy function; only the executor differs.
- Batch mode continues on per-file policy skips (not errors) unless configured otherwise.
