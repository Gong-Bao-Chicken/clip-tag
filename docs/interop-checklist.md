# Interop validation checklist (v0.1)

Manual verification matrix for metadata written by `clip-tag --write-metadata`.

## Fields written

| Field | ExifTool tag | Lightroom | Apple Photos | Capture One | Finder |
|-------|--------------|-----------|--------------|-------------|--------|
| XMP:Subject | `XMP-dc:Subject` | [ ] | [ ] | [ ] | [ ] |
| IPTC:Keywords | `IPTC:Keywords` | [ ] | [ ] | [ ] | [ ] |
| EXIF:XPKeywords | `EXIF:XPKeywords` | [ ] | [ ] | [ ] | [ ] |
| dc:subject | `XMP-dc:Subject` | [ ] | [ ] | [ ] | [ ] |

## Procedure

1. Pick a fixture image with empty keyword fields:
   ```bash
   exiftool -XMP-dc:Subject -IPTC:Keywords -EXIF:XPKeywords fixtures/corpus/sample.jpg
   ```
2. Tag and write metadata:
   ```bash
   cargo run -p clip-tag -- fixtures/corpus/sample.jpg --write-metadata --top-k 5
   ```
3. Confirm fields populated:
   ```bash
   exiftool -json -G1 -XMP-dc:Subject -IPTC:Keywords -EXIF:XPKeywords fixtures/corpus/sample.jpg
   ```
4. Open the same file in each target app and confirm keywords appear in the UI.

## Policy checks

| Scenario | Command | Expected |
|----------|---------|----------|
| Empty fields | `--write-metadata` | Tags written |
| Non-empty fields | `--write-metadata` (no force) | Skip, no mutation |
| Non-empty + force | `--write-metadata --force` | Overwrite |
| Dry run | `--dry-run --write-metadata` | Plan printed, no mutation |

## Notes

- Metadata I/O uses [ExifTool](https://exiftool.org/) (`brew install exiftool`).
- RAW sidecar writes are still deferred for a future release.
- Default tagging flags: `--threshold 0.01`, `--diversity-threshold 0.8` (see [models.md](models.md)).
