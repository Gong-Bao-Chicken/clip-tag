# ADR 004: Metadata field mapping contract

## Status

Accepted (v0.1)

## Context

Target applications read different metadata namespaces. clip-tag must write a consistent keyword set across fields for interoperability.

## Decision

For non-RAW images (JPEG, PNG, TIFF), v0.1 writes the **same normalized tag list** to each field:

| Canonical field | ExifTool name | Notes |
|-----------------|---------------|-------|
| `XmpSubject` | `XMP:Subject` | Bag of text values |
| `IptcKeywords` | `IPTC:Keywords` | Legacy IPTC keyword list |
| `ExifXpKeywords` | `EXIF:XPKeywords` | Windows shell / Finder |
| `DcSubject` | `dc:subject` | Dublin Core subject |

- Field order in plans is stable: XMP → IPTC → EXIF XP → dc.
- RAW formats: XMP sidecar only (documented; implemented after v0.1).

## Consequences

- `clip-tag-core::write::MetadataField` is the single enum for planners and executors.
- Interop validation (Phase D) uses this matrix as the checklist.
