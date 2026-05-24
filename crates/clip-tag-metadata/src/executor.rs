use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Mutex, OnceLock};

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

/// Long-lived ExifTool process driven via `-stay_open True -@ -`.
///
/// ExifTool startup is ~250 ms on macOS — fork+exec dominates wall time when
/// reading/writing thousands of files. The daemon keeps one process resident
/// and pipes per-file commands in.
struct ExifToolDaemon {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    counter: u64,
}

impl ExifToolDaemon {
    fn spawn() -> std::io::Result<Self> {
        let mut child = Command::new(EXIFTOOL)
            .arg("-stay_open")
            .arg("True")
            .arg("-@")
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        let stderr = child.stderr.take().expect("stderr");
        // Drain stderr so the daemon never blocks on a full pipe.
        std::thread::spawn(move || drain(stderr));
        Ok(Self {
            child,
            stdin,
            stdout,
            counter: 0,
        })
    }

    fn execute(&mut self, args: &[&str]) -> Result<String> {
        self.counter += 1;
        let n = self.counter;
        for arg in args {
            writeln!(self.stdin, "{arg}").map_err(|e| Error::Read(e.to_string()))?;
        }
        writeln!(self.stdin, "-execute{n}").map_err(|e| Error::Read(e.to_string()))?;
        self.stdin.flush().map_err(|e| Error::Read(e.to_string()))?;

        let sentinel = format!("{{ready{n}}}");
        let mut out = String::new();
        let mut line = String::new();
        loop {
            line.clear();
            let read = self
                .stdout
                .read_line(&mut line)
                .map_err(|e| Error::Read(e.to_string()))?;
            if read == 0 {
                return Err(Error::Read("exiftool daemon closed unexpectedly".into()));
            }
            if line.trim_end_matches(&['\r', '\n'][..]) == sentinel {
                break;
            }
            out.push_str(&line);
        }
        Ok(out)
    }
}

impl Drop for ExifToolDaemon {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "-stay_open\nFalse\n-execute");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

fn drain<R: Read>(mut r: R) {
    let mut buf = [0u8; 4096];
    while r.read(&mut buf).map(|n| n > 0).unwrap_or(false) {}
}

/// Process-wide daemon handle. `None` means the daemon failed to spawn and
/// callers should fall back to per-call `Command::new("exiftool")`.
static DAEMON: OnceLock<Option<Mutex<ExifToolDaemon>>> = OnceLock::new();

fn daemon() -> Option<&'static Mutex<ExifToolDaemon>> {
    DAEMON
        .get_or_init(|| match ExifToolDaemon::spawn() {
            Ok(d) => {
                tracing::debug!("exiftool stay-open daemon ready");
                Some(Mutex::new(d))
            }
            Err(e) => {
                tracing::warn!(
                    "exiftool stay-open daemon unavailable ({e}); falling back to per-call spawn",
                );
                None
            }
        })
        .as_ref()
}

fn run_exiftool(args: &[&str]) -> Result<String> {
    if let Some(d) = daemon() {
        let mut guard = d
            .lock()
            .map_err(|e| Error::Read(format!("exiftool daemon mutex poisoned: {e}")))?;
        return guard.execute(args);
    }

    let output = Command::new(EXIFTOOL)
        .args(args)
        .output()
        .map_err(|e| Error::Read(format!("failed to run exiftool: {e}")))?;
    if !output.status.success() {
        return Err(Error::Read(format!(
            "exiftool failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    String::from_utf8(output.stdout).map_err(|e| Error::Read(e.to_string()))
}

fn exiftool_json(path: &Path, tags: &[&str]) -> Result<serde_json::Value> {
    let path_str = path
        .to_str()
        .ok_or_else(|| Error::Read(format!("non-UTF8 path: {}", path.display())))?;

    let mut args: Vec<String> = Vec::with_capacity(tags.len() + 3);
    args.push("-json".into());
    args.push("-n".into());
    for tag in tags {
        args.push(format!("-{tag}"));
    }
    args.push(path_str.into());
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();

    let stdout = run_exiftool(&arg_refs)?;
    let rows: Vec<serde_json::Value> =
        serde_json::from_str(&stdout).map_err(|e| Error::Read(e.to_string()))?;
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

    let path_str = path
        .to_str()
        .ok_or_else(|| Error::Write(format!("non-UTF8 path: {}", path.display())))?;

    let mut written = std::collections::BTreeSet::new();
    let mut args: Vec<String> = Vec::new();
    args.push("-overwrite_original".into());

    for op in plan.operations() {
        for tag in field_tags(op.field) {
            if !written.insert(tag) {
                continue;
            }
            if op.overwrite {
                args.push(format!("-{tag}="));
            }
            for value in &op.values {
                args.push(format!("-{tag}={value}"));
            }
        }
    }

    args.push(path_str.into());
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();

    run_exiftool(&arg_refs).map_err(|e| Error::Write(e.to_string()))?;
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
