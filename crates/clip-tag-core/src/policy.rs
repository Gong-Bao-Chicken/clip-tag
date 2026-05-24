use serde::{Deserialize, Serialize};

use crate::write::{FieldSnapshot, MetadataField, WritePlan};
use crate::Result;

/// How metadata writes are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteMode {
    /// Add tags only when all target keyword fields are empty.
    #[default]
    EmptyOnly,
    /// Overwrite existing keyword values when `--force` is set.
    ForceOverwrite,
}

/// Inputs to the write policy engine (single decision point).
#[derive(Debug, Clone)]
pub struct WritePolicyInput {
    pub mode: WriteMode,
    pub force: bool,
    pub proposed_tags: Vec<String>,
    pub current: FieldSnapshot,
}

/// Planned write outcome before persistence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteDecision {
    Skip { reason: String },
    Apply(WritePlan),
}

pub fn plan_write(input: WritePolicyInput) -> Result<WriteDecision> {
    let tags = crate::normalize::normalize_tags(input.proposed_tags);

    if tags.is_empty() {
        return Ok(WriteDecision::Skip {
            reason: "no tags after normalization".into(),
        });
    }

    let fields = MetadataField::all();
    let any_nonempty = fields.iter().any(|f| input.current.get(*f).is_some());

    let allow_overwrite = input.force || input.mode == WriteMode::ForceOverwrite;

    if any_nonempty && !allow_overwrite {
        return Ok(WriteDecision::Skip {
            reason: "target keyword fields are not empty (use --force to overwrite)".into(),
        });
    }

    Ok(WriteDecision::Apply(WritePlan {
        tags,
        fields: fields.to_vec(),
        overwrite: allow_overwrite,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::FieldSnapshot;

    fn empty_snapshot() -> FieldSnapshot {
        FieldSnapshot::default()
    }

    #[test]
    fn empty_fields_allows_write() {
        let decision = plan_write(WritePolicyInput {
            mode: WriteMode::EmptyOnly,
            force: false,
            proposed_tags: vec!["Sunset".into()],
            current: empty_snapshot(),
        })
        .unwrap();

        assert!(matches!(decision, WriteDecision::Apply(_)));
    }

    #[test]
    fn nonempty_fields_skipped_without_force() {
        let mut current = FieldSnapshot::default();
        current.set(MetadataField::XmpSubject, vec!["existing".into()]);

        let decision = plan_write(WritePolicyInput {
            mode: WriteMode::EmptyOnly,
            force: false,
            proposed_tags: vec!["beach".into()],
            current,
        })
        .unwrap();

        assert!(matches!(decision, WriteDecision::Skip { .. }));
    }

    #[test]
    fn force_allows_overwrite() {
        let mut current = FieldSnapshot::default();
        current.set(MetadataField::IptcKeywords, vec!["old".into()]);

        let decision = plan_write(WritePolicyInput {
            mode: WriteMode::EmptyOnly,
            force: true,
            proposed_tags: vec!["new".into()],
            current,
        })
        .unwrap();

        match decision {
            WriteDecision::Apply(plan) => assert!(plan.overwrite),
            _ => panic!("expected apply"),
        }
    }
}
