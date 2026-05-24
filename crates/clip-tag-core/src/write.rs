use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Canonical metadata fields written by clip-tag (see ADR-004).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataField {
    XmpSubject,
    IptcKeywords,
    ExifXpKeywords,
    DcSubject,
}

impl MetadataField {
    pub const fn all() -> &'static [Self] {
        &[
            Self::XmpSubject,
            Self::IptcKeywords,
            Self::ExifXpKeywords,
            Self::DcSubject,
        ]
    }

    pub const fn xmp_name(self) -> &'static str {
        match self {
            Self::XmpSubject => "XMP:Subject",
            Self::IptcKeywords => "IPTC:Keywords",
            Self::ExifXpKeywords => "EXIF:XPKeywords",
            Self::DcSubject => "dc:subject",
        }
    }
}

/// Current values read from an image before planning writes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldSnapshot {
    values: BTreeMap<MetadataField, Vec<String>>,
}

impl FieldSnapshot {
    pub fn get(&self, field: MetadataField) -> Option<&[String]> {
        self.values.get(&field).map(Vec::as_slice)
    }

    pub fn set(&mut self, field: MetadataField, values: Vec<String>) {
        if values.is_empty() {
            self.values.remove(&field);
        } else {
            self.values.insert(field, values);
        }
    }
}

/// Immutable plan produced by the policy engine; executed by `clip-tag-metadata`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WritePlan {
    pub tags: Vec<String>,
    pub fields: Vec<MetadataField>,
    pub overwrite: bool,
}

/// One field update within a plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteOperation {
    pub field: MetadataField,
    pub values: Vec<String>,
    pub overwrite: bool,
}

impl WritePlan {
    pub fn operations(&self) -> Vec<WriteOperation> {
        self.fields
            .iter()
            .map(|field| WriteOperation {
                field: *field,
                values: self.tags.clone(),
                overwrite: self.overwrite,
            })
            .collect()
    }
}
