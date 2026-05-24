use std::path::Path;
use std::process::Command;

use clip_tag_core::write::{FieldSnapshot, MetadataField, WritePlan};

use crate::{Error, Result};

const EXIFTOOL: &str = "exiftool";

fn field_tags(field: MetadataField) -> &'static [&'static str] {
    match field {
        MetadataField::XmpSubject | MetadataField::DcSubject => &["XMP-dc:Subject"],
        MetadataField::IptcKeywords => &["IPTC:Keywords"],
        MetadataField::ExifXpKeywords => &["EXIF:XPKeywords"],
    }
}

fn json_keys(field: MetadataField) -> &'static [&'static str] {
    match field {
        MetadataField::XmpSubject | MetadataField::DcSubject => &["Subject", "XMP-dc:Subject"],
        MetadataField::IptcKeywords => &["Keywords", "IPTC:Keywords"],
        MetadataField::ExifXpKeywords => &["XPKeywords", "EXIF:XPKeywords"],
    }
}

fn exiftool_json(path: &Path, tags: &[&str]) -> Result<serde_json::Value> {
    let mut cmd = Command::new(EXIFTOOL);
    cmd.arg("-json").arg("-n");
    for tag in tags {
        cmd.arg(format!("-{tag}"));
    }
    cmd.arg(path);

    let output = cmd
        .output()
        .map_err(|e| Error::Read(format!("failed to run exiftool: {e}")))?;
    if !output.status.success() {
        return Err(Error::Read(format!(
            "exiftool read failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let rows: Vec<serde_json::Value> =
        serde_json::from_slice(&output.stdout).map_err(|e| Error::Read(e.to_string()))?;
    Ok(rows.into_iter().next().unwrap_or(serde_json::json!({})))
}

fn values_from_json(row: &serde_json::Value, tag: &str) -> Vec<String> {
    let Some(value) = row.get(tag) else {
        return vec![];
    };
    match value {
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        serde_json::Value::String(s) if !s.is_empty() => s
            .split(';')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_string)
            .collect(),
        _ => vec![],
    }
}

/// Read current keyword field values from an image.
pub fn read_field_snapshot(path: &Path) -> Result<FieldSnapshot> {
    let tags = ["XMP-dc:Subject", "IPTC:Keywords", "EXIF:XPKeywords"];
    let row = exiftool_json(path, &tags)?;
    let mut snapshot = FieldSnapshot::default();

    for field in MetadataField::all() {
        let mut merged = Vec::new();
        for key in json_keys(*field) {
            for value in values_from_json(&row, key) {
                if !merged.contains(&value) {
                    merged.push(value);
                }
            }
        }
        if !merged.is_empty() {
            snapshot.set(*field, merged);
        }
    }
    Ok(snapshot)
}

/// Apply a policy-approved [`WritePlan`] in-place via ExifTool.
pub fn execute_plan(path: &Path, plan: &WritePlan, dry_run: bool) -> Result<()> {
    if dry_run {
        return Ok(());
    }

    let mut written = std::collections::BTreeSet::new();
    let mut cmd = Command::new(EXIFTOOL);
    cmd.arg("-overwrite_original");

    for op in plan.operations() {
        for tag in field_tags(op.field) {
            if !written.insert(tag) {
                continue;
            }
            if op.overwrite {
                cmd.arg(format!("-{tag}="));
            }
            for value in &op.values {
                cmd.arg(format!("-{tag}={value}"));
            }
        }
    }

    cmd.arg(path);
    let output = cmd
        .output()
        .map_err(|e| Error::Write(format!("failed to run exiftool: {e}")))?;
    if !output.status.success() {
        return Err(Error::Write(format!(
            "exiftool write failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clip_tag_core::write::MetadataField;
    use image::{ImageBuffer, Rgb};
    use std::process::Command;
    use tempfile::tempdir;

    fn exiftool_installed() -> bool {
        Command::new(EXIFTOOL)
            .arg("-ver")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn write_blank_jpeg(path: &Path) {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(32, 32, |x, y| {
            Rgb([((x * 7) % 256) as u8, ((y * 11) % 256) as u8, 128])
        });
        img.save(path).unwrap();
    }

    #[test]
    fn write_and_read_keywords() {
        if !exiftool_installed() {
            eprintln!("skipping: exiftool not installed");
            return;
        }
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.jpg");
        write_blank_jpeg(&path);

        let plan = WritePlan {
            tags: vec!["sunset".into(), "beach".into()],
            fields: MetadataField::all().to_vec(),
            overwrite: true,
        };
        execute_plan(&path, &plan, false).unwrap();

        let snapshot = read_field_snapshot(&path).unwrap();
        assert!(snapshot.get(MetadataField::XmpSubject).is_some());
        assert!(snapshot.get(MetadataField::IptcKeywords).is_some());
    }

    #[test]
    fn dry_run_does_not_write() {
        if !exiftool_installed() {
            return;
        }
        let dir = tempdir().unwrap();
        let path = dir.path().join("dry.jpg");
        write_blank_jpeg(&path);

        let plan = WritePlan {
            tags: vec!["mountain".into()],
            fields: MetadataField::all().to_vec(),
            overwrite: true,
        };
        execute_plan(&path, &plan, true).unwrap();
        let snapshot = read_field_snapshot(&path).unwrap();
        assert!(snapshot.get(MetadataField::IptcKeywords).is_none());
    }
}
