use clip_tag_core::policy::{plan_write, WriteDecision, WriteMode, WritePolicyInput};
use clip_tag_core::write::{FieldSnapshot, MetadataField};

#[test]
fn dry_run_plan_matches_force_gate() {
    let mut current = FieldSnapshot::default();
    current.set(MetadataField::IptcKeywords, vec!["existing".into()]);

    let skip = plan_write(WritePolicyInput {
        mode: WriteMode::EmptyOnly,
        force: false,
        proposed_tags: vec!["beach".into()],
        current: current.clone(),
    })
    .unwrap();
    assert!(matches!(skip, WriteDecision::Skip { .. }));

    let apply = plan_write(WritePolicyInput {
        mode: WriteMode::EmptyOnly,
        force: true,
        proposed_tags: vec!["beach".into()],
        current,
    })
    .unwrap();
    assert!(matches!(apply, WriteDecision::Apply(_)));
}
