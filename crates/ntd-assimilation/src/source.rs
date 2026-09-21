#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use ntd_capsule::{sha256, Digest, SectionKind};
use ntd_ir::Graph;
use ntd_runtime::CapabilityDescriptor;

use crate::AssimilationError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LicenseRecord {
    pub spdx_id: String,
    pub notice: String,
}

impl LicenseRecord {
    pub fn new(
        spdx_id: impl Into<String>,
        notice: impl Into<String>,
    ) -> Result<Self, AssimilationError> {
        let spdx_id = spdx_id.into().trim().to_owned();
        if spdx_id.is_empty() {
            return Err(AssimilationError::InvalidProvenance);
        }
        Ok(Self {
            spdx_id,
            notice: notice.into(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceRecord {
    pub source_uri: String,
    pub source_digest: Digest,
    pub license: LicenseRecord,
    pub attribution: String,
}

impl ProvenanceRecord {
    pub fn validate(&self) -> Result<(), AssimilationError> {
        if self.source_uri.trim().is_empty() || self.license.spdx_id.trim().is_empty() {
            return Err(AssimilationError::InvalidProvenance);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcePackage {
    pub media_type: String,
    pub payload: Vec<u8>,
    pub provenance: ProvenanceRecord,
}

impl SourcePackage {
    pub fn new(
        media_type: impl Into<String>,
        payload: Vec<u8>,
        source_uri: impl Into<String>,
        license: LicenseRecord,
        attribution: impl Into<String>,
    ) -> Result<Self, AssimilationError> {
        let media_type = media_type.into().trim().to_owned();
        let source_uri = source_uri.into().trim().to_owned();
        if media_type.is_empty() || source_uri.is_empty() || payload.is_empty() {
            return Err(AssimilationError::InvalidSource);
        }
        let provenance = ProvenanceRecord {
            source_uri,
            source_digest: sha256(&payload),
            license,
            attribution: attribution.into(),
        };
        provenance.validate()?;
        Ok(Self {
            media_type,
            payload,
            provenance,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetKind {
    Capability,
    Intelligence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Discovery {
    pub importer_id: String,
    pub asset_kind: AssetKind,
    pub asset_id_hint: String,
}

impl Discovery {
    pub fn validate(&self) -> Result<(), AssimilationError> {
        if self.importer_id.trim().is_empty() || self.asset_id_hint.trim().is_empty() {
            return Err(AssimilationError::InvalidCandidate(
                "invalid discovery metadata".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAdapter {
    pub format: String,
    pub bytes: Vec<u8>,
}

impl NativeAdapter {
    pub fn validate(&self) -> Result<(), AssimilationError> {
        if self.format.trim().is_empty() || self.bytes.is_empty() {
            return Err(AssimilationError::InvalidCandidate(
                "empty native adapter".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeSection {
    pub kind: SectionKind,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegressionProbe {
    GraphRoundTrip,
    AdapterDigest(Digest),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegressionCase {
    pub name: String,
    pub probe: RegressionProbe,
}

impl RegressionCase {
    pub fn validate(&self) -> Result<(), AssimilationError> {
        if self.name.trim().is_empty() {
            return Err(AssimilationError::InvalidCandidate(
                "empty regression name".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeCandidate {
    Capability {
        asset_id: String,
        version: u32,
        descriptor: CapabilityDescriptor,
        adapter: NativeAdapter,
        regressions: Vec<RegressionCase>,
    },
    Intelligence {
        asset_id: String,
        version: u32,
        graph: Graph,
        sections: Vec<NativeSection>,
        regressions: Vec<RegressionCase>,
    },
}

impl NativeCandidate {
    pub fn asset_id(&self) -> &str {
        match self {
            Self::Capability { asset_id, .. } | Self::Intelligence { asset_id, .. } => asset_id,
        }
    }

    pub fn version(&self) -> u32 {
        match self {
            Self::Capability { version, .. } | Self::Intelligence { version, .. } => *version,
        }
    }

    pub fn kind(&self) -> AssetKind {
        match self {
            Self::Capability { .. } => AssetKind::Capability,
            Self::Intelligence { .. } => AssetKind::Intelligence,
        }
    }

    pub fn regressions(&self) -> &[RegressionCase] {
        match self {
            Self::Capability { regressions, .. } | Self::Intelligence { regressions, .. } => {
                regressions
            }
        }
    }

    pub fn validate(&self) -> Result<(), AssimilationError> {
        if self.asset_id().trim().is_empty() || self.version() == 0 {
            return Err(AssimilationError::InvalidCandidate(
                "invalid asset identity".into(),
            ));
        }

        let mut names = BTreeSet::new();
        for regression in self.regressions() {
            regression.validate()?;
            if !names.insert(regression.name.trim().to_owned()) {
                return Err(AssimilationError::InvalidCandidate(
                    "duplicate regression".into(),
                ));
            }
        }
        if names.is_empty() {
            return Err(AssimilationError::InvalidCandidate(
                "at least one regression is required".into(),
            ));
        }

        match self {
            Self::Capability {
                descriptor,
                adapter,
                ..
            } => {
                let mut descriptor = descriptor.clone();
                descriptor
                    .normalize()
                    .map_err(|error| AssimilationError::InvalidCandidate(format!("{error:?}")))?;
                adapter.validate()?;
            }
            Self::Intelligence {
                graph, sections, ..
            } => {
                graph
                    .validate()
                    .map_err(|error| AssimilationError::InvalidCandidate(format!("{error:?}")))?;
                let mut kinds = BTreeSet::new();
                for section in sections {
                    if section.bytes.is_empty()
                        || matches!(
                            section.kind,
                            SectionKind::Graph
                                | SectionKind::Provenance
                                | SectionKind::Signatures
                                | SectionKind::AssimilationLog
                        )
                        || !kinds.insert(section.kind as u16)
                    {
                        return Err(AssimilationError::InvalidCandidate(
                            "invalid native section set".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
