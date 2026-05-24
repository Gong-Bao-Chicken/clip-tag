use serde::{Deserialize, Serialize};

use crate::policy::WriteDecision;
use crate::write::WritePlan;

/// Human- and machine-readable report for `--dry-run`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DryRunReport {
    pub path: String,
    pub decision: DryRunDecision,
    pub operations: Vec<DryRunOperation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DryRunDecision {
    Skip { reason: String },
    WouldWrite,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DryRunOperation {
    pub field: String,
    pub values: Vec<String>,
    pub overwrite: bool,
}

pub fn report_for_path(path: impl Into<String>, decision: &WriteDecision) -> DryRunReport {
    let path = path.into();
    match decision {
        WriteDecision::Skip { reason } => DryRunReport {
            path,
            decision: DryRunDecision::Skip {
                reason: reason.clone(),
            },
            operations: vec![],
        },
        WriteDecision::Apply(plan) => DryRunReport {
            path,
            decision: DryRunDecision::WouldWrite,
            operations: plan_to_operations(plan),
        },
    }
}

fn plan_to_operations(plan: &WritePlan) -> Vec<DryRunOperation> {
    plan.operations()
        .into_iter()
        .map(|op| DryRunOperation {
            field: op.field.xmp_name().to_string(),
            values: op.values,
            overwrite: op.overwrite,
        })
        .collect()
}
