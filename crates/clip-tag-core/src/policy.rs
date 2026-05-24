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
    /// Merge proposed tags with existing tags, then rewrite the canonical set.
    Merge,
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
    let existing = crate::normalize::normalize_tags(
        MetadataField::all()
            .iter()
            .flat_map(|field| input.current.get(*field).unwrap_or(&[]).iter().cloned())
            .collect::<Vec<String>>(),
    );

    if tags.is_empty() {
        return Ok(WriteDecision::Skip {
            reason: "no tags after normalization".into(),
        });
    }

    let fields = MetadataField::all();
    let any_nonempty = fields.iter().any(|f| input.current.get(*f).is_some());

    let allow_overwrite = input.force || input.mode == WriteMode::ForceOverwrite;
    let is_merge = input.mode == WriteMode::Merge && !input.force;

    if is_merge {
        let mut merged = existing;
        for tag in tags {
            if !merged.contains(&tag) {
                merged.push(tag);
            }
        }
        if merged.is_empty() {
            return Ok(WriteDecision::Skip {
                reason: "no tags after merge".into(),
            });
        }
        return Ok(WriteDecision::Apply(WritePlan {
            tags: merged,
            fields: fields.to_vec(),
            overwrite: true,
        }));
    }

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

    #[test]
    fn merge_mode_combines_existing_and_new_tags() {
        let mut current = FieldSnapshot::default();
        current.set(MetadataField::XmpSubject, vec!["beach".into()]);

        let decision = plan_write(WritePolicyInput {
            mode: WriteMode::Merge,
            force: false,
            proposed_tags: vec!["sunset".into(), "beach".into()],
            current,
        })
        .unwrap();

        match decision {
            WriteDecision::Apply(plan) => {
                assert_eq!(plan.tags, vec!["beach".to_string(), "sunset".to_string()]);
                assert!(plan.overwrite);
            }
            _ => panic!("expected apply"),
        }
    }
}
