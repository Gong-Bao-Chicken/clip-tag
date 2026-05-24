# ADR 001: `--force` semantics

## Status

Accepted (v0.1)

## Context

Users need a safe default that never destroys existing keywords, while power users must be able to replace stale or incorrect metadata.

## Decision

- Without `--force`, the write policy engine **never overwrites** existing values in any target keyword field.
- With `--force`, the engine may **overwrite** all mapped keyword fields (`XMP:Subject`, `IPTC:Keywords`, `EXIF:XPKeywords`, `dc:subject`) with the normalized tag set from the current run.
- `--force` does **not** enable network access, RAW in-place writes, or deletion of unrelated metadata blocks in v0.1.

## Consequences

- Implementation lives solely in `clip-tag-core::policy`; metadata writers execute the resulting plan verbatim.
- CLI must require both `--write-metadata` and `--force` for destructive updates.
