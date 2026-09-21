#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use ntd_capsule::{decode_graph, encode_graph, sha256};

use crate::{AssimilationError, NativeCandidate, RegressionProbe};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxReport {
    pub isolated: bool,
    pub network_used: bool,
    pub external_write_used: bool,
    pub passed_regressions: Vec<String>,
}

pub trait ForgeSandbox {
    fn validate(&mut self, candidate: &NativeCandidate)
        -> Result<SandboxReport, AssimilationError>;
}

#[derive(Debug, Default)]
pub struct NativeValidationSandbox;

impl ForgeSandbox for NativeValidationSandbox {
    fn validate(
        &mut self,
        candidate: &NativeCandidate,
    ) -> Result<SandboxReport, AssimilationError> {
        candidate.validate()?;
        let mut passed = Vec::new();

        for regression in candidate.regressions() {
            match (&regression.probe, candidate) {
                (RegressionProbe::GraphRoundTrip, NativeCandidate::Intelligence { graph, .. }) => {
                    let encoded = encode_graph(graph).map_err(|error| {
                        AssimilationError::SandboxRejected(format!("{error:?}"))
                    })?;
                    let decoded = decode_graph(&encoded).map_err(|error| {
                        AssimilationError::SandboxRejected(format!("{error:?}"))
                    })?;
                    if decoded != *graph {
                        return Err(AssimilationError::SandboxRejected(
                            "graph round trip drift".into(),
                        ));
                    }
                }
                (
                    RegressionProbe::AdapterDigest(expected),
                    NativeCandidate::Capability { adapter, .. },
                ) => {
                    if sha256(&adapter.bytes) != *expected {
                        return Err(AssimilationError::SandboxRejected(
                            "adapter digest regression".into(),
                        ));
                    }
                }
                _ => {
                    return Err(AssimilationError::SandboxRejected(
                        "regression probe does not match candidate".into(),
                    ))
                }
            }
            passed.push(regression.name.trim().to_owned());
        }

        let unique = passed.iter().cloned().collect::<BTreeSet<_>>();
        if unique.len() != passed.len() {
            return Err(AssimilationError::SandboxRejected(
                "duplicate regression result".into(),
            ));
        }

        Ok(SandboxReport {
            isolated: true,
            network_used: false,
            external_write_used: false,
            passed_regressions: passed,
        })
    }
}
