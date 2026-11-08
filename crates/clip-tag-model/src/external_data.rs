//! Fold ONNX external-data initializers into the model file itself.
//!
//! Background
//! ----------
//!
//! Several RuteNL OpenCLIP ONNX exports ship their weights in a sidecar file
//! (`text.onnx.data`, `visual.onnx.data`). The `.onnx` file then holds
//! `TensorProto`s with `data_location = EXTERNAL` and a
//! `StringStringEntryProto` pointing at the sidecar.
//!
//! ONNX Runtime's CoreML execution provider (and a few others) hits an
//! assertion when it materializes such initializers during graph
//! partitioning:
//!
//! ```text
//! initializer.cc:45 Initializer::Initializer(const TensorProto&, const std::filesystem::path&)
//! !model_path.empty() was false. model_path must not be empty.
//! ```
//!
//! That's because the partitioner constructs an interior sub-model and loses
//! the original on-disk path needed to resolve the external file. The fix is
//! to inline the external data into the `.onnx` file itself before ORT ever
//! sees it.
//!
//! Approach
//! --------
//!
//! Rather than depending on a generated protobuf crate, this module
//! manually walks the protobuf wire format. We only inspect three nested
//! fields:
//!
//! - `ModelProto.graph` (field 7)
//! - `GraphProto.initializer` (field 5, repeated `TensorProto`)
//! - `TensorProto.{raw_data, external_data, data_location}` (fields 9, 13, 14)
//!
//! Every other field — opset metadata, the node list, type info, anything
//! ONNX adds in a future version — is copied through as opaque bytes. The
//! rewriter therefore stays correct across ONNX schema additions.

use std::fs;
use std::path::Path;

use crate::{Error, Result};

// --- ONNX field numbers (onnx.proto3) ---

const MODEL_PROTO_GRAPH: u64 = 7;
const GRAPH_PROTO_INITIALIZER: u64 = 5;
const TENSOR_PROTO_RAW_DATA: u64 = 9;
const TENSOR_PROTO_EXTERNAL_DATA: u64 = 13;
const TENSOR_PROTO_DATA_LOCATION: u64 = 14;
const STRING_STRING_ENTRY_KEY: u64 = 1;
const STRING_STRING_ENTRY_VALUE: u64 = 2;

// --- Protobuf wire types ---

const WIRE_VARINT: u8 = 0;
const WIRE_FIXED64: u8 = 1;
const WIRE_LENGTH_DELIMITED: u8 = 2;
const WIRE_FIXED32: u8 = 5;

// --- ONNX DataLocation enum ---

const DATA_LOCATION_EXTERNAL: u64 = 1;

/// Returns true if any initializer in the model at `onnx_path` uses external
/// data storage.
pub fn has_external_data(onnx_path: &Path) -> Result<bool> {
    let bytes = read_file(onnx_path)?;
    scan_model_for_external(&bytes)
}

/// Rewrite `src_onnx` into `dst_onnx` with all external initializer data
/// inlined as `raw_data`. The `.onnx.data` files referenced by the source
/// are read from `src_onnx`'s directory.
///
/// No-op (just a byte copy) if the source has no external initializers.
pub fn fold_onnx_file(src_onnx: &Path, dst_onnx: &Path) -> Result<()> {
    let bytes = read_file(src_onnx)?;
    let src_dir = src_onnx
        .parent()
        .ok_or_else(|| Error::Load(format!("no parent dir for {}", src_onnx.display())))?;
    let folded = rewrite_model_proto(&bytes, src_dir)?;
    write_file(dst_onnx, &folded)
}

// ---------------------------------------------------------------------------
// scan: detect external_data without modifying anything
// ---------------------------------------------------------------------------

fn scan_model_for_external(bytes: &[u8]) -> Result<bool> {
    let mut walker = Walker::new(bytes);
    while let Some(field) = walker.next_field()? {
        if field.number == MODEL_PROTO_GRAPH && field.wire_type == WIRE_LENGTH_DELIMITED {
            if scan_graph_for_external(field.bytes)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn scan_graph_for_external(bytes: &[u8]) -> Result<bool> {
    let mut walker = Walker::new(bytes);
    while let Some(field) = walker.next_field()? {
        if field.number == GRAPH_PROTO_INITIALIZER && field.wire_type == WIRE_LENGTH_DELIMITED {
            if tensor_is_external(field.bytes)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn tensor_is_external(bytes: &[u8]) -> Result<bool> {
    let mut walker = Walker::new(bytes);
    while let Some(field) = walker.next_field()? {
        if field.number == TENSOR_PROTO_DATA_LOCATION
            && field.wire_type == WIRE_VARINT
            && field.varint == DATA_LOCATION_EXTERNAL
        {
            return Ok(true);
        }
    }
    Ok(false)
}

// ---------------------------------------------------------------------------
// rewrite: inline external initializers
// ---------------------------------------------------------------------------

fn rewrite_model_proto(bytes: &[u8], src_dir: &Path) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut walker = Walker::new(bytes);
    while let Some(field) = walker.next_field()? {
        if field.number == MODEL_PROTO_GRAPH && field.wire_type == WIRE_LENGTH_DELIMITED {
            let new_graph = rewrite_graph_proto(field.bytes, src_dir)?;
            emit_length_delimited(&mut out, MODEL_PROTO_GRAPH, &new_graph);
        } else {
            out.extend_from_slice(field.raw);
        }
    }
    Ok(out)
}

fn rewrite_graph_proto(bytes: &[u8], src_dir: &Path) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut walker = Walker::new(bytes);
    while let Some(field) = walker.next_field()? {
        if field.number == GRAPH_PROTO_INITIALIZER && field.wire_type == WIRE_LENGTH_DELIMITED {
            let new_tensor = rewrite_tensor_proto(field.bytes, src_dir)?;
            emit_length_delimited(&mut out, GRAPH_PROTO_INITIALIZER, &new_tensor);
        } else {
            out.extend_from_slice(field.raw);
        }
    }
    Ok(out)
}

fn rewrite_tensor_proto(bytes: &[u8], src_dir: &Path) -> Result<Vec<u8>> {
    // Pass 1: discover whether this tensor uses external data and where.
    let mut is_external = false;
    let mut location: Option<String> = None;
    let mut offset: usize = 0;
    let mut length: Option<usize> = None;

    let mut walker = Walker::new(bytes);
    while let Some(field) = walker.next_field()? {
        match (field.number, field.wire_type) {
            (TENSOR_PROTO_DATA_LOCATION, WIRE_VARINT) => {
                if field.varint == DATA_LOCATION_EXTERNAL {
                    is_external = true;
                }
            }
            (TENSOR_PROTO_EXTERNAL_DATA, WIRE_LENGTH_DELIMITED) => {
                let (k, v) = parse_string_string_entry(field.bytes)?;
                match k.as_str() {
                    "location" => location = Some(v),
                    "offset" => offset = v.parse().unwrap_or(0),
                    "length" => length = v.parse().ok(),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    if !is_external {
        return Ok(bytes.to_vec());
    }

    let location = location.ok_or_else(|| {
        Error::Load("external-data tensor missing `location`".into())
    })?;
    let data_path = src_dir.join(&location);
    let data = read_file(&data_path)?;
    let length = length.unwrap_or_else(|| data.len().saturating_sub(offset));
    let end = offset
        .checked_add(length)
        .ok_or_else(|| Error::Load("external-data slice overflow".into()))?;
    if end > data.len() {
        return Err(Error::Load(format!(
            "external slice {}..{} exceeds {} size {}",
            offset,
            end,
            data_path.display(),
            data.len()
        )));
    }
    let payload = &data[offset..end];

    // Pass 2: re-emit, dropping the three fields we're replacing and
    // appending an inline raw_data field at the end.
    let mut out = Vec::with_capacity(bytes.len() + payload.len());
    let mut walker = Walker::new(bytes);
    while let Some(field) = walker.next_field()? {
        match field.number {
            TENSOR_PROTO_RAW_DATA
            | TENSOR_PROTO_EXTERNAL_DATA
            | TENSOR_PROTO_DATA_LOCATION => { /* skip */ }
            _ => out.extend_from_slice(field.raw),
        }
    }
    emit_length_delimited(&mut out, TENSOR_PROTO_RAW_DATA, payload);
    Ok(out)
}

fn parse_string_string_entry(bytes: &[u8]) -> Result<(String, String)> {
    let mut key = String::new();
    let mut value = String::new();
    let mut walker = Walker::new(bytes);
    while let Some(field) = walker.next_field()? {
        if field.wire_type != WIRE_LENGTH_DELIMITED {
            continue;
        }
        let text = std::str::from_utf8(field.bytes)
            .map_err(|e| Error::Load(format!("non-UTF8 external_data entry: {e}")))?
            .to_string();
        match field.number {
            STRING_STRING_ENTRY_KEY => key = text,
            STRING_STRING_ENTRY_VALUE => value = text,
            _ => {}
        }
    }
    Ok((key, value))
}

// ---------------------------------------------------------------------------
// protobuf wire-format helpers
// ---------------------------------------------------------------------------

struct Walker<'a> {
    bytes: &'a [u8],
    pos: usize,
}

struct Field<'a> {
    number: u64,
    wire_type: u8,
    /// Decoded varint, when `wire_type == WIRE_VARINT`. Otherwise 0.
    varint: u64,
    /// Length-delimited payload bytes (without tag/length), when applicable.
    /// Empty for other wire types.
    bytes: &'a [u8],
    /// The entire field bytes, tag + value, suitable for passthrough writes.
    raw: &'a [u8],
}

impl<'a> Walker<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn next_field(&mut self) -> Result<Option<Field<'a>>> {
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let field_start = self.pos;
        let (tag, n) = read_varint(&self.bytes[self.pos..])?;
        self.pos += n;
        let number = tag >> 3;
        let wire_type = (tag & 0x7) as u8;

        let (varint, payload) = match wire_type {
            WIRE_VARINT => {
                let (v, n) = read_varint(&self.bytes[self.pos..])?;
                self.pos += n;
                (v, &[][..])
            }
            WIRE_FIXED64 => {
                self.advance(8)?;
                (0, &[][..])
            }
            WIRE_LENGTH_DELIMITED => {
                let (len, n) = read_varint(&self.bytes[self.pos..])?;
                self.pos += n;
                let len = len as usize;
                self.advance(len)?;
                let payload = &self.bytes[self.pos - len..self.pos];
                (0, payload)
            }
            WIRE_FIXED32 => {
                self.advance(4)?;
                (0, &[][..])
            }
            other => {
                return Err(Error::Load(format!(
                    "unsupported protobuf wire type {other}"
                )));
            }
        };

        let raw = &self.bytes[field_start..self.pos];
        Ok(Some(Field {
            number,
            wire_type,
            varint,
            bytes: payload,
            raw,
        }))
    }

    fn advance(&mut self, n: usize) -> Result<()> {
        if self.pos + n > self.bytes.len() {
            return Err(Error::Load("truncated protobuf field".into()));
        }
        self.pos += n;
        Ok(())
    }
}

fn read_varint(bytes: &[u8]) -> Result<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        result |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return Err(Error::Load("protobuf varint too long".into()));
        }
    }
    Err(Error::Load("protobuf varint truncated".into()))
}

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let b = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn emit_length_delimited(out: &mut Vec<u8>, field_number: u64, payload: &[u8]) {
    let tag = (field_number << 3) | u64::from(WIRE_LENGTH_DELIMITED);
    write_varint(out, tag);
    write_varint(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| Error::Load(format!("read {}: {e}", path.display())))
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| Error::Load(format!("mkdir {}: {e}", parent.display())))?;
    }
    fs::write(path, bytes).map_err(|e| Error::Load(format!("write {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tag(field: u64, wire: u8) -> Vec<u8> {
        let mut out = Vec::new();
        write_varint(&mut out, (field << 3) | u64::from(wire));
        out
    }

    fn varint(v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        write_varint(&mut out, v);
        out
    }

    fn len_delim(field: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = make_tag(field, WIRE_LENGTH_DELIMITED);
        out.extend(varint(payload.len() as u64));
        out.extend_from_slice(payload);
        out
    }

    fn make_string_string_entry(key: &str, value: &str) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(len_delim(STRING_STRING_ENTRY_KEY, key.as_bytes()));
        out.extend(len_delim(STRING_STRING_ENTRY_VALUE, value.as_bytes()));
        out
    }

    fn make_external_tensor(name: &str, location: &str, offset: usize, length: usize) -> Vec<u8> {
        // TensorProto fields: name (field 8, string), data_location (14, varint),
        // external_data (13, repeated StringStringEntryProto), dims (1, repeated int64).
        let mut out = Vec::new();
        // dims = [2]
        out.extend(make_tag(1, WIRE_VARINT));
        out.extend(varint(2));
        // name
        out.extend(len_delim(8, name.as_bytes()));
        // external_data: location/offset/length entries
        out.extend(len_delim(
            TENSOR_PROTO_EXTERNAL_DATA,
            &make_string_string_entry("location", location),
        ));
        out.extend(len_delim(
            TENSOR_PROTO_EXTERNAL_DATA,
            &make_string_string_entry("offset", &offset.to_string()),
        ));
        out.extend(len_delim(
            TENSOR_PROTO_EXTERNAL_DATA,
            &make_string_string_entry("length", &length.to_string()),
        ));
        // data_location = EXTERNAL
        out.extend(make_tag(TENSOR_PROTO_DATA_LOCATION, WIRE_VARINT));
        out.extend(varint(DATA_LOCATION_EXTERNAL));
        out
    }

    fn make_internal_tensor(name: &str, raw: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        // dims = [<raw.len() / 4>]
        out.extend(make_tag(1, WIRE_VARINT));
        out.extend(varint((raw.len() / 4) as u64));
        // name
        out.extend(len_delim(8, name.as_bytes()));
        // raw_data
        out.extend(len_delim(TENSOR_PROTO_RAW_DATA, raw));
        out
    }

    fn make_model(initializers: &[Vec<u8>]) -> Vec<u8> {
        let mut graph = Vec::new();
        // unrelated graph field to validate passthrough (name, field 2)
        graph.extend(len_delim(2, b"graph-name"));
        for init in initializers {
            graph.extend(len_delim(GRAPH_PROTO_INITIALIZER, init));
        }

        let mut model = Vec::new();
        // ir_version (field 1, varint)
        model.extend(make_tag(1, WIRE_VARINT));
        model.extend(varint(10));
        // producer_name (field 2, string)
        model.extend(len_delim(2, b"test"));
        // graph
        model.extend(len_delim(MODEL_PROTO_GRAPH, &graph));
        model
    }

    #[test]
    fn varint_roundtrip() {
        for v in [0u64, 1, 127, 128, 16383, 16384, 1_000_000, u64::MAX / 2] {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            let (decoded, n) = read_varint(&buf).unwrap();
            assert_eq!(decoded, v, "varint {v} round-trip");
            assert_eq!(n, buf.len(), "varint length consistent");
        }
    }

    #[test]
    fn walker_passes_through_unknown_fields() {
        // Three fields: varint, length-delimited string, fixed32.
        let mut bytes = Vec::new();
        bytes.extend(make_tag(1, WIRE_VARINT));
        bytes.extend(varint(42));
        bytes.extend(len_delim(2, b"hello"));
        bytes.extend(make_tag(3, WIRE_FIXED32));
        bytes.extend([1, 2, 3, 4]);

        let mut walker = Walker::new(&bytes);
        let mut rebuilt = Vec::new();
        while let Some(field) = walker.next_field().unwrap() {
            rebuilt.extend_from_slice(field.raw);
        }
        assert_eq!(rebuilt, bytes, "byte-for-byte passthrough");
    }

    #[test]
    fn detect_external_data_in_synthetic_model() {
        let external_tensor = make_external_tensor("w", "data.bin", 0, 16);
        let model = make_model(&[external_tensor]);
        assert!(scan_model_for_external(&model).unwrap());

        let internal_tensor = make_internal_tensor("w", &[0u8; 16]);
        let plain = make_model(&[internal_tensor]);
        assert!(!scan_model_for_external(&plain).unwrap());
    }

    #[test]
    fn fold_inlines_external_data_and_drops_location_fields() {
        let temp = tempfile::tempdir().unwrap();
        let data_bytes: Vec<u8> = (0..32u8).collect();
        std::fs::write(temp.path().join("weights.bin"), &data_bytes).unwrap();

        let tensor = make_external_tensor("w", "weights.bin", 8, 16);
        let model = make_model(&[tensor]);
        let src_path = temp.path().join("test.onnx");
        std::fs::write(&src_path, &model).unwrap();

        let dst_path = temp.path().join("folded.onnx");
        fold_onnx_file(&src_path, &dst_path).unwrap();

        let folded = std::fs::read(&dst_path).unwrap();
        assert!(
            !scan_model_for_external(&folded).unwrap(),
            "folded model should not advertise external data"
        );

        // Walk the folded TensorProto and verify raw_data == the slice we
        // expected from the external file.
        let mut model_walker = Walker::new(&folded);
        let graph_bytes = loop {
            let f = model_walker.next_field().unwrap().unwrap();
            if f.number == MODEL_PROTO_GRAPH {
                break f.bytes.to_vec();
            }
        };
        let mut graph_walker = Walker::new(&graph_bytes);
        let init_bytes = loop {
            let f = graph_walker.next_field().unwrap().unwrap();
            if f.number == GRAPH_PROTO_INITIALIZER {
                break f.bytes.to_vec();
            }
        };
        let mut tensor_walker = Walker::new(&init_bytes);
        let mut got_raw: Option<Vec<u8>> = None;
        while let Some(f) = tensor_walker.next_field().unwrap() {
            assert_ne!(
                f.number, TENSOR_PROTO_EXTERNAL_DATA,
                "external_data field should be removed"
            );
            assert_ne!(
                f.number, TENSOR_PROTO_DATA_LOCATION,
                "data_location field should be removed"
            );
            if f.number == TENSOR_PROTO_RAW_DATA {
                got_raw = Some(f.bytes.to_vec());
            }
        }
        let raw = got_raw.expect("folded tensor must have raw_data");
        assert_eq!(raw, data_bytes[8..24], "raw_data slice matches offset+length");
    }

    #[test]
    fn fold_is_a_noop_for_internal_tensors() {
        let temp = tempfile::tempdir().unwrap();
        let model = make_model(&[make_internal_tensor("w", &[5u8; 12])]);
        let src = temp.path().join("plain.onnx");
        let dst = temp.path().join("plain.folded.onnx");
        std::fs::write(&src, &model).unwrap();
        fold_onnx_file(&src, &dst).unwrap();
        let folded = std::fs::read(&dst).unwrap();
        assert_eq!(folded, model, "no external data → byte-identical output");
    }

    #[test]
    fn has_external_data_reports_false_when_no_external() {
        let temp = tempfile::tempdir().unwrap();
        let model = make_model(&[make_internal_tensor("w", &[0u8; 8])]);
        let path = temp.path().join("plain.onnx");
        std::fs::write(&path, model).unwrap();
        assert!(!has_external_data(&path).unwrap());
    }
}
